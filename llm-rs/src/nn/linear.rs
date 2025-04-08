use super::{NeuralNetwork, macros::*, unique};
use crate::{Blob, Context, Tensor};
use digit_layout::types;
use gemm::{Parallelism::Rayon, gemm};
use mem_rearrange::Rearranging;
use std::iter::zip;
use tensor::rw_rc::RwRc;

pub struct Linear {
    w: RwRc<Tensor<Blob>>,
    b: Option<RwRc<Tensor<Blob>>>,
    x: Option<RwRc<Tensor<Blob>>>,
}

impl NeuralNetwork for Linear {
    type Init = (RwRc<Tensor<Blob>>, Option<RwRc<Tensor<Blob>>>);

    fn init(init: Self::Init, _ctx: &mut Context) -> Self {
        let (weight, bias) = init;
        Self {
            w: weight,
            b: bias,
            x: None,
        }
    }

    fn forward(
        &mut self,
        inputs: impl IntoIterator<Item = RwRc<Tensor<Blob>>>,
        ctx: &mut Context,
    ) -> Vec<RwRc<Tensor<Blob>>> {
        destruct!([x] = inputs);
        self.x.replace(x);
        let Self { w, b, x } = self;

        let x = x.as_ref().unwrap().read();
        let w = w.read();
        dims!([batch_size, seq_len, _] = x);
        dims!([d, _] = w);
        let mut y = ctx.tensor(x.dt(), &[batch_size, seq_len, d]);

        ctx.bench(|| {
            forward(
                y.as_deref_mut().merge(0, 2),
                x.as_deref().merge(0, 2),
                w.as_deref(),
                b.as_ref().map(|t| t.read().as_deref()),
            )
        });

        vec![y.share()]
    }

    fn backward(
        &mut self,
        inputs: impl IntoIterator<Item = RwRc<Tensor<Blob>>>,
        ctx: &mut Context,
    ) -> Vec<RwRc<Tensor<Blob>>> {
        destruct!([dy] = inputs);
        let Self { w, b, x } = self;

        let x = x.take().unwrap();
        let x = x.read();
        let dw = ctx.write_gradient("w", w);
        let mut dx = Tensor::contiguous_of(x).map(Blob::new_zeroed);
        let db = b.as_ref().map(|b| ctx.write_gradient("b", b));
        ctx.bench(|| {
            backward(
                dx.as_deref_mut().merge(0, 2),
                dw.write().as_deref_mut(),
                db.as_ref().map(|t| t.write().as_deref_mut()),
                dy.read().as_deref().merge(0, 2),
                x.as_deref().merge(0, 2),
                w.read().as_deref(),
            )
        });

        w.release();
        if let Some(b) = &b {
            b.release()
        }

        vec![dx.share()]
    }
}

fn forward(
    mut y: Tensor<&mut [u8]>,
    x: Tensor<&[u8]>,
    weight: Tensor<&[u8]>,
    bias: Option<Tensor<&[u8]>>,
) {
    let dt = unique(&[y.dt(), x.dt(), weight.dt()]).unwrap();
    assert_eq!(dt, types::F32);

    dims!([m, n] = y);
    dims!([m_, k] = x);
    dims!([n_, k_] = weight);

    assert_eq!(m, m_);
    assert_eq!(k, k_);
    assert_eq!(n, n_);

    if let Some(bias) = &bias {
        assert_eq!(bias.dt(), dt);
        dims!([n__] = bias);
        assert_eq!(n_, n__);

        unsafe {
            Rearranging::new(
                y.layout(),
                &bias.layout().tile_be(0, &[1, n]).broadcast(0, m),
                dt.nbytes(),
            )
            .unwrap()
            .launch(y.get_mut().as_mut_ptr(), bias.get().as_ptr())
        }
    }

    unsafe {
        gemm::<f32>(
            m,
            n,
            k,
            y.mut_ptr(),
            1,
            n as isize,
            bias.is_some(),
            x.ptr(),
            1,
            k as isize,
            weight.ptr(),
            k as isize,
            1,
            1.,
            1.,
            false,
            false,
            false,
            Rayon(0),
        )
    }
}

fn backward(
    mut dx: Tensor<&mut [u8]>,
    mut dw: Tensor<&mut [u8]>,
    db: Option<Tensor<&mut [u8]>>,
    dy: Tensor<&[u8]>,
    x: Tensor<&[u8]>,
    w: Tensor<&[u8]>,
) {
    let dt = unique(&[dx.dt(), dw.dt(), dy.dt(), x.dt(), w.dt()]).unwrap();
    assert_eq!(dt, types::F32);

    dims!([m, n] = dx);
    dims!([m_, k] = dy);
    dims!([k_, n_] = w);
    assert_eq!(m, m_);
    assert_eq!(n, n_);
    assert_eq!(k, k_);

    unsafe {
        gemm::<f32>(
            m,
            n,
            k,
            dx.mut_ptr(),
            1,
            n as _,
            false,
            dy.ptr(),
            1,
            k as _,
            w.ptr(),
            1,
            n as _,
            1.,
            1.,
            false,
            false,
            false,
            Rayon(0),
        )
    };

    dims!([m, n] = dw);
    dims!([k, m_] = dy);
    dims!([k_, n_] = x);
    assert_eq!(m, m_);
    assert_eq!(n, n_);
    assert_eq!(k, k_);

    unsafe {
        gemm::<f32>(
            m,
            n,
            k,
            dw.mut_ptr(),
            1,
            n as _,
            false,
            dy.ptr(),
            m as _,
            1,
            x.ptr(),
            1,
            n as _,
            1.,
            1.,
            false,
            false,
            false,
            Rayon(0),
        )
    }

    if let Some(mut db) = db {
        assert_eq!(db.dt(), types::F32);

        dims!([n, d] = dy);
        dims!([d_] = db);
        assert_eq!(d, d_);

        let db = db.vector_mut::<f32>();
        for i in 0..n {
            let dy = dy.as_deref().index(&[i]).vector::<f32>();
            for (db, dy) in zip(&mut *db, dy) {
                *db += dy
            }
        }
    }
}
