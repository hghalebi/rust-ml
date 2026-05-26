use rust_ml_neuron::{Dataset, LearningRate, TinyNeuron};

fn average_loss(neuron: &TinyNeuron, dataset: &Dataset) -> Result<f64, rust_ml_neuron::Error> {
    let total = dataset
        .examples()
        .iter()
        .map(|example| neuron.loss(example).map(|loss| loss.value()))
        .sum::<Result<f64, _>>()?;

    Ok(total / dataset.len() as f64)
}

fn main() -> Result<(), rust_ml_neuron::Error> {
    let dataset = Dataset::and_gate()?;
    let mut neuron = TinyNeuron::lesson_seed()?;
    let rate = LearningRate::new(0.8)?;

    println!(
        "epoch 0 average loss = {:.4}",
        average_loss(&neuron, &dataset)?
    );

    for epoch in 1..=20 {
        neuron.train_epoch(&dataset, rate)?;

        if epoch == 1 || epoch == 5 || epoch == 10 || epoch == 20 {
            println!(
                "epoch {epoch} average loss = {:.4}",
                average_loss(&neuron, &dataset)?
            );
        }
    }

    Ok(())
}
