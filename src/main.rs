use f32;
use ndarray::prelude::*;
use ndarray::{Array1, Array2, ShapeBuilder};
// use num_traits::{Float, Integer};
use num::{Float, Integer};
use rand::distributions::{Distribution, Uniform};
use rand::rngs::ThreadRng;
use usize;

#[derive(Debug)]
enum TrainingType {
    Hebbian,
    Storkey,
}

#[derive(Debug)]
struct HopfieldNet<T> {
    s: Array1<T>, // State
    w: Array2<T>, // Weights
    rng: Uniform<usize>,
}
trait New {
    fn new(n: usize) -> Self;
}

trait Update {
    fn update(&mut self, rand_gen: &mut ThreadRng);
}

impl<T: Float> New for HopfieldNet<T> {
    fn new(n: usize) -> Self {
        HopfieldNet {
            s: Array1::<T>::ones(n.f()), // Yay for column major!
            w: Array2::<T>::zeros((n, n).f()),
            rng: Uniform::from(0..n),
        }
    }
}

impl<T: Float> Update for HopfieldNet<T> {
    fn update(&mut self, rand_gen: &mut ThreadRng) {
        println!("Before starting: s: {:?} w: {:?}", self.s, self.w);
        let i = self.rng.sample(rand_gen);
        let n_value = self.w.slice(s![.., i]).dot(&self.s).tanh();
        let mut s_slice = self.s.get_mut(i).unwrap(); // Live dangerously because we _know_ i can't be out of bounds
        *s_slice = n_value;
        println!("calculated value: {:?}", n_value);
        println!("Post updated: s: {:?} w: {:?}", self.s, self.w);
    }
}

impl<T: Integer> Update for HopfieldNet<T> {
    fn update(&mut self, rand_gen: &mut ThreadRng) {
        let i = self.rng.sample(rand_gen);
    }
}

fn energy<T>(net: &HopfieldNet<T>) -> f32 {
    let n = net.s.len();
    let mut e: f32 = -0.5 * &net.s.dot(&net.w).dot(&net.s.t());
    e += &net.w.column(1).dot(&net.s.t());
    e
}

fn settle<T>(net: &mut HopfieldNet<T>, n_iter: u32, rng: &mut ThreadRng) {
    let iters = 1..=n_iter;
    iters.for_each(|i| {
        net.update(rng);
        println!("Iteration: {}\nEnergy: {:.2}", i, energy(&net));
    });
}

fn associate<T>(net: &mut HopfieldNet<T>, pattern: Array1<f32>, n_iter: u32, rng: &mut ThreadRng) {
    net.s = pattern;
    settle(net, n_iter, rng);
}

fn train_net<T>(net: &mut HopfieldNet<T>, patterns: Array2<f32>, training_method: TrainingType) {
    match training_method {
        TrainingType::Hebbian => train_hebbian(net, patterns),
        TrainingType::Storkey => train_storkey(net, patterns),
    };
}

fn train_hebbian<T>(net: &mut HopfieldNet<T>, patterns: Array2<f32>) {
    let n = net.s.len();
    let p = patterns.ncols();
    let p_div = p as f32;
    // TODO: Convert from direct index-based access to high level .map, etc functions
    for i in 0..n {
        for j in (i + 1)..n {
            let mut s = 0.;
            for mu in 0..p {
                s += patterns[[i, mu]] * patterns[[j, mu]];
            }
            s = s / p_div;
            net.w[[i, j]] = s;
            net.w[[j, i]] = s;
        }
    }
}

fn train_storkey<T>(net: &mut HopfieldNet<T>, patterns: Array2<f32>) {
    let n = net.s.len();
    let p = patterns.ncols();
    let n_div = n as f32;
    for i in 0..n {
        for j in (i + 1)..n {
            for mu in 0..p {
                let mut s = patterns[[i, mu]] * patterns[[j, mu]];
                s -= patterns[[i, mu]] * h_fn(j, i, mu, n, &net.w, &patterns);
                s -= h_fn(i, j, mu, n, &net.w, &patterns) * patterns[[j, mu]];
                s *= 1. / n_div;
                net.w[[i, j]] += s;
                net.w[[j, i]] += s;
            }
        }
    }
}

fn h_fn(
    i: usize,
    j: usize,
    mu: usize,
    n: usize,
    weights: &Array2<f32>,
    patterns: &Array2<f32>,
) -> f32 {
    let mut res = 0.;
    for k in 0..n {
        if k != i && k != j {
            res += weights[[i, k]] * patterns[[k, mu]];
        }
    }
    res
}

fn main() {
    println!("Starting up!");
    let mut rng = rand::thread_rng();
    let mut net = HopfieldNet::new(4);
    net.s = array![0.21, 0.003413, 0.3723829382, 0.9858328];
    println!("Net is now: {:?}", net);
    net.update(&mut rng);
}
