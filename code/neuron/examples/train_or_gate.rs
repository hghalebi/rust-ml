use rust_ml_neuron::{
    Bias, Dataset, LearningRate, Neuron, Sgd, Weight, average_loss, train_epochs,
};

fn main() {
    let dataset = Dataset::or_gate();
    let mut neuron = Neuron::new(Weight(0.0), Weight(0.0), Bias(0.0));
    let optimizer = Sgd::new(LearningRate(0.8));

    let before = f64::from(average_loss(&neuron, &dataset));
    let metrics = train_epochs(&mut neuron, &dataset, optimizer, 500);
    let after = f64::from(average_loss(&neuron, &dataset));

    println!("average loss: before={before:.4}, after={after:.4}");

    for example in dataset.iter() {
        let prediction = f64::from(neuron.predict(example.x1, example.x2));
        println!(
            "x=({:.0}, {:.0}) target={:.0} prediction={prediction:.4}",
            example.x1.0, example.x2.0, example.target.0
        );
    }

    if let Some(last_epoch) = metrics.last() {
        println!(
            "last epoch={} average_loss={:.4}",
            last_epoch.epoch,
            f64::from(last_epoch.average_loss)
        );
    }
}
