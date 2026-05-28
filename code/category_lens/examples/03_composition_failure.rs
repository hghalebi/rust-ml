use rust_ml_category_lens::{MapName, ObjectName, TypedMap, TypedObject};

fn main() -> Result<(), rust_ml_category_lens::Error> {
    let feature_vector = TypedObject::new(ObjectName::try_from("FeatureVector")?);
    let pre_activation = TypedObject::new(ObjectName::try_from("PreActivation")?);
    let prediction = TypedObject::new(ObjectName::try_from("Prediction")?);
    let loss = TypedObject::new(ObjectName::try_from("Loss")?);

    let raw_score = TypedMap::new(
        MapName::try_from("raw_score")?,
        feature_vector,
        pre_activation,
    );
    let squared_error = TypedMap::new(MapName::try_from("squared_error")?, prediction, loss);

    match &raw_score >> &squared_error {
        Ok(composite) => println!("{composite}"),
        Err(error) => println!("{error}"),
    }

    Ok(())
}
