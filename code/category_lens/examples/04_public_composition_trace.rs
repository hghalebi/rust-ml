use rust_ml_category_lens::{
    CompositionTrace, CompositionTraceVisibility, Error, MapName, ObjectName,
    PublicCompositionTrace, ReviewedCompositionTrace, TypedMap, TypedObject,
};

fn reviewed_trace(
    visibility: CompositionTraceVisibility,
) -> Result<ReviewedCompositionTrace, Error> {
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

    Ok(ReviewedCompositionTrace::new(trace, visibility))
}

fn main() -> Result<(), Error> {
    let public_trace = PublicCompositionTrace::from_reviewed_trace(reviewed_trace(
        CompositionTraceVisibility::Public,
    )?)?;

    println!("public map count = {}", public_trace.maps().len());
    println!("public composite = {}", public_trace.composite()?);

    let private_trace = PublicCompositionTrace::from_reviewed_trace(reviewed_trace(
        CompositionTraceVisibility::Private,
    )?);

    if let Err(error) = private_trace {
        println!("blocked from public composition trace: {error}");
    }

    Ok(())
}
