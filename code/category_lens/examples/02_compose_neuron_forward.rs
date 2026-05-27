use rust_ml_category_lens::{CompositionTrace, MapName, ObjectName, TypedMap, TypedObject};

fn main() -> Result<(), rust_ml_category_lens::Error> {
    let feature_vector = TypedObject::new(ObjectName::try_from("FeatureVector")?);
    let pre_activation = TypedObject::new(ObjectName::try_from("PreActivation")?);
    let prediction = TypedObject::new(ObjectName::try_from("Prediction")?);
    let loss = TypedObject::new(ObjectName::try_from("Loss")?);

    let trace = CompositionTrace::from_maps([
        TypedMap::new(
            MapName::try_from("raw_score")?,
            feature_vector,
            pre_activation.clone(),
        ),
        TypedMap::new(
            MapName::try_from("sigmoid")?,
            pre_activation,
            prediction.clone(),
        ),
        TypedMap::new(MapName::try_from("squared_error")?, prediction, loss),
    ])?;

    println!("{}", trace.composite()?);
    Ok(())
}
