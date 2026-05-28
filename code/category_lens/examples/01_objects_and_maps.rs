use rust_ml_category_lens::{MapName, ObjectName, TypedMap, TypedObject};

fn main() -> Result<(), rust_ml_category_lens::Error> {
    let feature_vector = TypedObject::new(ObjectName::try_from("FeatureVector")?);
    let pre_activation = TypedObject::new(ObjectName::try_from("PreActivation")?);
    let raw_score = TypedMap::new(
        MapName::try_from("raw_score")?,
        feature_vector,
        pre_activation,
    );

    println!("{raw_score}");
    Ok(())
}
