use super::{NeuralNetwork, macros::*, unique};
use crate::{Blob, Context, Tensor};
use digit_layout::types;
use itertools::izip;
use std::{iter::zip, slice::from_raw_parts_mut};
use tensor::rw_rc::RwRc;

pub struct Attention {
    nh: usize,
    x: Option<RwRc<Tensor<Blob>>>,
    att: Option<Tensor<Blob>>,
}

impl NeuralNetwork for Attention {
    type Init = usize;

    fn init(init: Self::Init, _ctx: &mut Context) -> Self {
        Self {
            nh: init,
            x: None,
            att: None,
        }
    }

    fn forward(
        &mut self,
        inputs: impl IntoIterator<Item = RwRc<Tensor<Blob>>>,
        ctx: &mut Context,
    ) -> Vec<RwRc<Tensor<Blob>>> {
        destruct!([x] = inputs);
        self.x.replace(x);
        let Self { nh, x, .. } = self;

        let x = x.as_ref().unwrap().read();
        dims!([batch_size, n_seq, d3] = x);

        let d = d3 / 3;
        let mut y = ctx.tensor_zeroed(x.dt(), &[batch_size, n_seq, d]);
        let mut preatt = ctx.tensor_zeroed(x.dt(), &[batch_size, *nh, n_seq, n_seq]);
        let mut att = ctx.tensor_zeroed(x.dt(), &[batch_size, *nh, n_seq, n_seq]);

        ctx.bench(|| {
            forward(
                y.as_deref_mut(),
                preatt.as_deref_mut(),
                att.as_deref_mut(),
                x.as_deref(),
            )
        });

        self.att.replace(att);

        vec![y.share()]
    }

    fn backward(
        &mut self,
        inputs: impl IntoIterator<Item = RwRc<Tensor<Blob>>>,
        ctx: &mut Context,
    ) -> Vec<RwRc<Tensor<Blob>>> {
        destruct!([dy] = inputs);
        let Self { x, att, .. } = self;

        let x = x.take().unwrap();
        let x = x.read();
        let mut dx = Tensor::contiguous_of(x).map(Blob::new_zeroed);

        let att = att.take().unwrap();
        let mut dpreatt = Tensor::contiguous_of(&att).map(Blob::new_zeroed);
        let mut datt = Tensor::contiguous_of(&att).map(Blob::new_zeroed);

        ctx.bench(|| {
            backward(
                dx.as_deref_mut(),
                dpreatt.as_deref_mut(),
                datt.as_deref_mut(),
                dy.read().as_deref(),
                x.as_deref(),
                att.as_deref(),
            )
        });

        vec![dx.share()]
    }
}

fn forward(
    mut y: Tensor<&mut [u8]>,
    mut preatt: Tensor<&mut [u8]>,
    mut att: Tensor<&mut [u8]>,
    x: Tensor<&[u8]>,
) {
    let dt = unique(&[y.dt(), preatt.dt(), att.dt(), x.dt()]).unwrap();
    assert_eq!(dt, types::F32);

    dims!([batch_size_0, n_seq_0, d] = y);
    dims!([batch_size_1, n_seq_1, d3] = x);
    dims!([batch_size_2, nh_0, n_seq_2, n_seq_3] = preatt);
    dims!([batch_size_3, nh_1, n_seq_4, n_seq_5] = att);

    let batch_size = unique(&[batch_size_0, batch_size_1, batch_size_2, batch_size_3]).unwrap();
    let n_seq = unique(&[n_seq_0, n_seq_1, n_seq_2, n_seq_3, n_seq_4, n_seq_5]).unwrap();
    let nh = unique(&[nh_0, nh_1]).unwrap();
    let d = unique(&[d, d3 / 3]).unwrap();
    let dh = d / nh;
    let scale = (dh as f32).powf(-0.5);

    for b in 0..batch_size {
        let qkv = x.as_deref().index(&[b]);
        for t in 0..n_seq {
            let q = qkv.as_deref().index(&[t]).vector::<f32>();
            let y = y.as_deref_mut().index(&[b, t]).vector_mut::<f32>();

            for h in 0..nh {
                let y = &mut y[h * dh..][..dh];
                let q = &q[h * dh..][..dh];

                let preatt = preatt.as_deref_mut().index(&[b, h, t]).vector_mut::<f32>();
                let att = att.as_deref_mut().index(&[b, h, t]).vector_mut::<f32>();
                let (preatt, _) = preatt.split_at_mut(t + 1);
                let (att, tail) = att.split_at_mut(t + 1);

                // pass 1: calculate query dot key and maxval
                let mut max = f32::NEG_INFINITY;
                for (t_, val) in preatt.iter_mut().enumerate() {
                    let k = &qkv.as_deref().index(&[t_]).vector::<f32>()[d..][h * dh..][..dh];
                    *val = zip(q, k).map(|(&q, &k)| q * k).sum::<f32>() * scale;
                    if *val > max {
                        max = *val
                    }
                }

                // pass 2: calculate the exp and keep track of sum
                let mut expsum = 0.;
                for (att, preatt) in zip(&mut *att, preatt) {
                    *att = (*preatt - max).exp();
                    expsum += *att
                }
                let expsum_inv = 1. / expsum;

                // pass 3: normalize to get the softmax
                for val in &mut *att {
                    *val *= expsum_inv
                }
                tail.fill(0.);

                // pass 4: accumulate weighted values into the output of attention
                y.fill(0.);
                for (t_, val) in att.iter_mut().enumerate() {
                    let v = &qkv.as_deref().index(&[t_]).vector::<f32>()[d * 2..][h * dh..][..dh];
                    for (y, v) in zip(&mut *y, v) {
                        *y += *val * v
                    }
                }
            }
        }
    }
}

fn backward(
    mut dx: Tensor<&mut [u8]>,
    mut dpreatt: Tensor<&mut [u8]>,
    mut datt: Tensor<&mut [u8]>,
    dy: Tensor<&[u8]>,
    x: Tensor<&[u8]>,
    att: Tensor<&[u8]>,
) {
    let dt = unique(&[dx.dt(), dpreatt.dt(), datt.dt(), dy.dt(), x.dt(), att.dt()]).unwrap();
    assert_eq!(dt, types::F32);

    dims!([batch_size_0, n_seq_0, d3_0] = dx);
    dims!([batch_size_1, nh_0, n_seq_1, n_seq_2] = dpreatt);
    dims!([batch_size_2, nh_1, n_seq_3, n_seq_4] = datt);
    dims!([batch_size_3, n_seq_5, d_0] = dy);
    dims!([batch_size_4, n_seq_6, d3_1] = x);
    dims!([batch_size_5, nh_2, n_seq_7, n_seq_8] = att);

    let batch_size = unique(&[
        batch_size_0,
        batch_size_1,
        batch_size_2,
        batch_size_3,
        batch_size_4,
        batch_size_5,
    ])
    .unwrap();
    let n_seq = unique(&[
        n_seq_0, n_seq_1, n_seq_2, n_seq_3, n_seq_4, n_seq_5, n_seq_6, n_seq_7, n_seq_8,
    ])
    .unwrap();
    let nh = unique(&[nh_0, nh_1, nh_2]).unwrap();
    let d = unique(&[d3_0 / 3, d3_1 / 3, d_0]).unwrap();

    let dh = d / nh;
    let scale = (dh as f32).powf(-0.5);

    for b in 0..batch_size {
        for t in 0..n_seq {
            for h in 0..nh {
                let mut dx = dx.as_deref_mut().index(&[b]);
                let x = x.as_deref().index(&[b]);

                let dpreatt = dpreatt.as_deref_mut().index(&[b, h, t]).vector_mut::<f32>();
                let datt = datt.as_deref_mut().index(&[b, h, t]).vector_mut::<f32>();
                let dy = &dy.as_deref().index(&[b, t]).vector::<f32>()[h * dh..][..dh];
                let att = att.as_deref().index(&[b, h, t]).vector::<f32>();

                for t_ in 0..=t {
                    let dqkv = dx.as_deref_mut().index(&[t_]).vector_mut::<f32>();
                    let qkv = x.as_deref().index(&[t_]).vector::<f32>();

                    let dv = &mut dqkv[2 * d..][h * dh..][..dh];
                    let v = &qkv[2 * d..][h * dh..][..dh];
                    let datt = &mut datt[t_];
                    let att = att[t_];

                    for (dv, v, dy) in izip!(&mut *dv, v, dy) {
                        *datt += v * dy;
                        *dv += att * dy;
                    }
                }
                for t_ in 0..=t {
                    for t__ in 0..=t {
                        let indicator = if t_ == t__ { 1. } else { 0. };
                        dpreatt[t__] += att[t_] * (indicator - att[t__]) * datt[t_];
                    }
                }

                let dqkv = dx.as_deref_mut().merge(0, 2).vector_mut::<f32>();
                let qkv = x.as_deref().merge(0, 2).vector::<f32>();

                let dq =
                    unsafe { from_raw_parts_mut(dqkv[t * 3 * d..][h * dh..].as_mut_ptr(), dh) };
                let q = &qkv[t * 3 * d..][h * dh..][..dh];
                for t in 0..=t {
                    let dk = &mut dqkv[(t * 3 + 1) * d..][h * dh..][..dh];
                    let k = &qkv[(t * 3 + 1) * d..][h * dh..][..dh];
                    let dpreatt = dpreatt[t];

                    for (dq, q, dk, k) in izip!(&mut *dq, q, dk, k) {
                        *dq += k * dpreatt * scale;
                        *dk += q * dpreatt * scale;
                    }
                }
            }
        }
    }
}
