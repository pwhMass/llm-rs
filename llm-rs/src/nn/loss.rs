use super::{NeuralNetwork, macros::*, unique};
use crate::{Blob, Context, Tensor};
use digit_layout::types;
use std::iter::zip;
use tensor::rw_rc::RwRc;

pub struct Loss {
    n_voc: usize,
    targets: Option<RwRc<Tensor<Blob>>>,
    probs: Option<Tensor<Blob>>,
}

impl NeuralNetwork for Loss {
    type Init = usize;

    fn init(init: Self::Init, _ctx: &mut Context) -> Self {
        Self {
            n_voc: init,
            targets: None,
            probs: None,
        }
    }

    fn forward(
        &mut self,
        inputs: impl IntoIterator<Item = RwRc<Tensor<Blob>>>,
        ctx: &mut Context,
    ) -> Vec<RwRc<Tensor<Blob>>> {
        destruct!([logits, targets] = inputs);
        self.targets.replace(targets);
        let Self {
            n_voc: nvoc,
            targets,
            ..
        } = self;

        let targets = targets.as_ref().unwrap().read();
        let logits = logits.read();

        let mut probs = Tensor::contiguous_of(logits).map(Blob::new);
        softmax(probs.as_deref_mut(), logits.as_deref(), *nvoc);

        let mut losses = ctx.tensor(probs.dt(), &targets.shape());
        crossentropy(losses.as_deref_mut(), probs.as_deref(), targets.as_deref());

        self.probs.replace(probs);
        vec![losses.share()]
    }

    fn backward(
        &mut self,
        inputs: impl IntoIterator<Item = RwRc<Tensor<Blob>>>,
        _ctx: &mut Context,
    ) -> Vec<RwRc<Tensor<Blob>>> {
        destruct!([dlosses] = inputs);
        let Self { targets, probs, .. } = self;

        let probs = probs.take().unwrap();
        let mut dlogits = Tensor::contiguous_of(&probs).map(Blob::new_zeroed);

        backward(
            dlogits.as_deref_mut(),
            dlosses.read().as_deref(),
            probs.as_deref(),
            targets.take().unwrap().read().as_deref(),
        );

        vec![dlogits.share()]
    }
}

fn softmax(mut y: Tensor<&mut [u8]>, x: Tensor<&[u8]>, mask: usize) {
    let dt = unique(&[y.dt(), x.dt()]).unwrap();
    assert_eq!(dt, types::F32);

    dims!([batch_size, n_seq, n_voc] = y);
    dims!([batch_size_, n_seq_, n_voc_] = x);
    assert_eq!(batch_size, batch_size_);
    assert_eq!(n_seq, n_seq_);
    assert_eq!(n_voc, n_voc_);

    for b in 0..batch_size {
        for t in 0..n_seq {
            let y = y.as_deref_mut().index(&[b, t]).vector_mut::<f32>();
            let x = x.as_deref().index(&[b, t]).vector::<f32>();

            let (y, tail) = y.split_at_mut(mask);
            let x = &x[..mask];

            let max = x.iter().max_by(|a, b| f32::total_cmp(a, b)).unwrap();
            let mut expsum = 0.;
            for (y, &x) in zip(&mut *y, x) {
                *y = (x - max).exp();
                expsum += *y
            }

            for y in y {
                *y /= expsum
            }
            tail.fill(0.)
        }
    }
}

fn crossentropy(mut losses: Tensor<&mut [u8]>, probs: Tensor<&[u8]>, targets: Tensor<&[u8]>) {
    assert_eq!(unique(&[losses.dt(), probs.dt()]).unwrap(), types::F32);
    assert_eq!(targets.dt(), types::U16);

    dims!([batch_size_0, n_seq_0] = losses);
    dims!([batch_size_1, n_seq_1, _] = probs);
    dims!([batch_size_2, n_seq_2] = targets);

    let batch_size = unique(&[batch_size_0, batch_size_1, batch_size_2]).unwrap();
    let n_seq = unique(&[n_seq_0, n_seq_1, n_seq_2]).unwrap();

    for b in 0..batch_size {
        for t in 0..n_seq {
            let losses = losses.as_deref_mut().index(&[b, t]).scalar_mut::<f32>();
            let probs = probs.as_deref().index(&[b, t]).vector::<f32>();
            let target = targets.as_deref().index(&[b, t]).scalar::<u16>();
            *losses = -probs[*target as usize].ln()
        }
    }
}

fn backward(
    mut dlogits: Tensor<&mut [u8]>,
    dlosses: Tensor<&[u8]>,
    probs: Tensor<&[u8]>,
    targets: Tensor<&[u8]>,
) {
    let dt = unique(&[dlogits.dt(), dlosses.dt(), probs.dt()]).unwrap();
    assert_eq!(dt, types::F32);
    assert_eq!(targets.dt(), types::U16);

    dims!([batch_size_0, n_seq_0, n_voc_0] = dlogits);
    dims!([batch_size_1, n_seq_1] = dlosses);
    dims!([batch_size_2, n_seq_2, n_voc_1] = probs);
    dims!([batch_size_3, n_seq_3] = targets);

    let batch_size = unique(&[batch_size_0, batch_size_1, batch_size_2, batch_size_3]).unwrap();
    let n_seq = unique(&[n_seq_0, n_seq_1, n_seq_2, n_seq_3]).unwrap();
    let _ = unique(&[n_voc_0, n_voc_1]).unwrap();

    for b in 0..batch_size {
        for t in 0..n_seq {
            let dlogits = dlogits.as_deref_mut().index(&[b, t]).vector_mut::<f32>();
            let probs = probs.as_deref().index(&[b, t]).vector::<f32>();
            let dloss = *dlosses.as_deref().index(&[b, t]).scalar::<f32>();
            let ix = *targets.as_deref().index(&[b, t]).scalar::<u16>() as usize;
            for (i, (dlogit, prob)) in zip(dlogits, probs).enumerate() {
                let indicator = if i == ix { 1. } else { 0. };
                *dlogit += (prob - indicator) * dloss
            }
        }
    }
}
