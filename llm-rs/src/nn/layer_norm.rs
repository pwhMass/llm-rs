use super::{NeuralNetwork, macros::*, unique};
use crate::{Blob, Context, Tensor};
use digit_layout::types;
use itertools::izip;
use tensor::rw_rc::RwRc;

pub struct LayerNorm {
    w: RwRc<Tensor<Blob>>,
    b: RwRc<Tensor<Blob>>,
    x: Option<RwRc<Tensor<Blob>>>,
    mean: Option<Tensor<Blob>>,
    rstd: Option<Tensor<Blob>>,
}

impl NeuralNetwork for LayerNorm {
    type Init = [RwRc<Tensor<Blob>>; 2];

    fn init(init: Self::Init, _ctx: &mut Context) -> Self {
        let [scalar, bias] = init;
        Self {
            w: scalar,
            b: bias,
            x: None,
            mean: None,
            rstd: None,
        }
    }

    fn forward(
        &mut self,
        inputs: impl IntoIterator<Item = RwRc<Tensor<Blob>>>,
        ctx: &mut Context,
    ) -> Vec<RwRc<Tensor<Blob>>> {
        destruct!([x] = inputs);
        self.x.replace(x);
        let Self { w, b, x, .. } = self;

        let x = x.as_ref().unwrap().read();
        dims!([batch_size, n_seq, d] = x);

        let mut y = ctx.tensor(x.dt(), &[batch_size, n_seq, d]);
        let mut mean = ctx.tensor(x.dt(), &[batch_size, n_seq]);
        let mut rstd = ctx.tensor(x.dt(), &[batch_size, n_seq]);

        ctx.bench(|| {
            forward(
                y.as_deref_mut(),
                mean.as_deref_mut(),
                rstd.as_deref_mut(),
                x.as_deref(),
                w.read().as_deref(),
                b.read().as_deref(),
            )
        });

        self.mean.replace(mean);
        self.rstd.replace(rstd);

        vec![y.share()]
    }

    fn backward(
        &mut self,
        inputs: impl IntoIterator<Item = RwRc<Tensor<Blob>>>,
        ctx: &mut Context,
    ) -> Vec<RwRc<Tensor<Blob>>> {
        destruct!([dy] = inputs);
        let Self {
            w,
            b,
            x,
            mean,
            rstd,
        } = self;

        let x = x.take().unwrap();
        let x = x.read();
        let mut dx = Tensor::contiguous_of(x).map(Blob::new_zeroed);

        let dw = ctx.write_gradient("w", w);
        let db = ctx.write_gradient("b", b);
        ctx.bench(|| {
            backward(
                dx.as_deref_mut(),
                dw.write().as_deref_mut(),
                db.write().as_deref_mut(),
                dy.read().as_deref(),
                x.as_deref(),
                w.read().as_deref(),
                mean.take().unwrap().as_deref(),
                rstd.take().unwrap().as_deref(),
            )
        });

        w.release();
        b.release();

        vec![dx.share()]
    }
}

fn forward(
    mut y: Tensor<&mut [u8]>,
    mut mean: Tensor<&mut [u8]>,
    mut rstd: Tensor<&mut [u8]>,
    x: Tensor<&[u8]>,
    scalar: Tensor<&[u8]>,
    bias: Tensor<&[u8]>,
) {
    let dt = unique(&[y.dt(), mean.dt(), rstd.dt(), x.dt(), scalar.dt(), bias.dt()]).unwrap();
    assert_eq!(dt, types::F32);

    dims!([batch_size_0, n_seq_0, d_0] = y);
    dims!([batch_size_1, n_seq_1, d_1] = x);
    dims!([batch_size_2, n_seq_2] = mean);
    dims!([batch_size_3, n_seq_3] = rstd);
    dims!([d_2] = scalar);
    dims!([d_3] = bias);

    let batch_size = unique(&[batch_size_0, batch_size_1, batch_size_2, batch_size_3]).unwrap();
    let n_seq = unique(&[n_seq_0, n_seq_1, n_seq_2, n_seq_3]).unwrap();
    let d = unique(&[d_0, d_1, d_2, d_3]).unwrap();

    for b in 0..batch_size {
        for t in 0..n_seq {
            let out = y.as_deref_mut().index(&[b, t]).vector_mut::<f32>();

            let inp = x.as_deref().index(&[b, t]).vector::<f32>();
            let scalar = scalar.vector::<f32>();
            let bias = bias.vector::<f32>();

            const EPSILON: f32 = 1e-5;
            let mut sum = 0.;
            let mut sum2 = 0.;
            for &x in inp {
                sum += x;
                sum2 += x * x;
            }
            let mean_ = sum / (d as f32);
            let rstd_ = (sum2 / (d as f32) - mean_ * mean_ + EPSILON).powf(-0.5);

            *mean.as_deref_mut().index(&[b, t]).scalar_mut() = mean_;
            *rstd.as_deref_mut().index(&[b, t]).scalar_mut() = rstd_;

            for (out, scalar, bias, x) in izip!(&mut *out, scalar, bias, inp) {
                *out = (rstd_ * (x - mean_)) * scalar + bias
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn backward(
    mut dx: Tensor<&mut [u8]>,
    mut dw: Tensor<&mut [u8]>,
    mut db: Tensor<&mut [u8]>,
    dy: Tensor<&[u8]>,
    x: Tensor<&[u8]>,
    w: Tensor<&[u8]>,
    mean: Tensor<&[u8]>,
    rstd: Tensor<&[u8]>,
) {
    let dt = unique(&[
        dx.dt(),
        dw.dt(),
        db.dt(),
        dy.dt(),
        x.dt(),
        w.dt(),
        mean.dt(),
        rstd.dt(),
    ])
    .unwrap();
    assert_eq!(dt, types::F32);

    dims!([batch_size_0, n_seq_0, d_0] = dx);
    dims!([batch_size_1, n_seq_1, d_1] = dy);
    dims!([batch_size_2, n_seq_2, d_2] = x);
    dims!([d_3] = dw);
    dims!([d_4] = db);
    dims!([d_5] = w);
    dims!([batch_size_3, n_seq_3] = mean);
    dims!([batch_size_4, n_seq_4] = rstd);

    let batch_size = unique(&[
        batch_size_0,
        batch_size_1,
        batch_size_2,
        batch_size_3,
        batch_size_4,
    ])
    .unwrap();
    let n_seq = unique(&[n_seq_0, n_seq_1, n_seq_2, n_seq_3, n_seq_4]).unwrap();
    let d = unique(&[d_0, d_1, d_2, d_3, d_4, d_5]).unwrap();

    for b in 0..batch_size {
        for t in 0..n_seq {
            let dx = dx.as_deref_mut().index(&[b, t]).vector_mut::<f32>();
            let dw = dw.vector_mut::<f32>();
            let db = db.vector_mut::<f32>();
            let dy = dy.as_deref().index(&[b, t]).vector::<f32>();
            let x = x.as_deref().index(&[b, t]).vector::<f32>();
            let w = w.vector::<f32>();
            let mean = *mean.as_deref().index(&[b, t]).scalar::<f32>();
            let rstd = *rstd.as_deref().index(&[b, t]).scalar::<f32>();

            let mut dnorm_mean = 0.;
            let mut dnorm_norm_mean = 0.;
            for (dy, x, w) in izip!(dy, x, w) {
                let norm = (x - mean) * rstd;
                let dnorm = w * dy;
                dnorm_mean += dnorm;
                dnorm_norm_mean += norm * dnorm;
            }
            dnorm_mean /= d as f32;
            dnorm_norm_mean /= d as f32;

            for (dx, dw, db, dy, x, w) in izip!(dx, dw, db, dy, x, w) {
                let norm = (x - mean) * rstd;
                let dnorm = w * dy;
                *db += dy;
                *dw += norm * dy;
                *dx += rstd * (dnorm - dnorm_mean - norm * dnorm_norm_mean)
            }
        }
    }
}
