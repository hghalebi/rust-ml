//! A tiny category-theory lens for the Rust ML lessons.
//!
//! This crate keeps the vocabulary concrete:
//! - an object is a meaningful typed space such as `FeatureVector`
//! - a map is a checked transformation such as `FeatureVector -> PreActivation`
//! - composition is legal only when the target object of one map matches the
//!   source object of the next map
//!
//! Raw learner text enters through explicit `TryFrom` adapters. Public teaching
//! APIs then move through semantic values such as [`ObjectName`], [`MapName`],
//! [`TypedObject`], [`TypedMap`], and [`CompositionTrace`].

pub mod error;

use std::{fmt, ops::Shr};

use error::CategoryLensError;

pub use error::CategoryLensError as Error;

struct CheckedName(String);

impl CheckedName {
    fn into_string(self) -> String {
        self.0
    }
}

fn checked_name(
    role: &'static str,
    value: impl Into<String>,
) -> Result<CheckedName, CategoryLensError> {
    let value = value.into();
    let trimmed = value.trim();

    if trimmed.is_empty() {
        return Err(CategoryLensError::empty_name(role));
    }

    if trimmed.chars().any(char::is_control) {
        return Err(CategoryLensError::invalid_name(role, value));
    }

    Ok(CheckedName(trimmed.to_owned()))
}

/// Learner-visible name for a category object.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObjectName(String);

impl ObjectName {
    fn from_raw(value: impl Into<String>) -> Result<Self, CategoryLensError> {
        Ok(Self(checked_name("object", value)?.into_string()))
    }
}

impl TryFrom<&str> for ObjectName {
    type Error = CategoryLensError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::from_raw(value)
    }
}

impl fmt::Display for ObjectName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Learner-visible name for a typed map.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MapName(String);

impl MapName {
    fn from_raw(value: impl Into<String>) -> Result<Self, CategoryLensError> {
        Ok(Self(checked_name("map", value)?.into_string()))
    }

    fn composed(first: &MapName, second: &MapName) -> Result<Self, CategoryLensError> {
        Self::from_raw(format!("{second} after {first}"))
    }
}

impl TryFrom<&str> for MapName {
    type Error = CategoryLensError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::from_raw(value)
    }
}

impl fmt::Display for MapName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// A meaningful space of values in the learner's model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedObject {
    name: ObjectName,
}

impl TypedObject {
    /// Creates an object from a checked semantic name.
    pub fn new(name: ObjectName) -> Self {
        Self { name }
    }

    /// Returns the object name.
    pub fn name(&self) -> &ObjectName {
        &self.name
    }
}

impl fmt::Display for TypedObject {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.name, formatter)
    }
}

/// A typed transformation from one object to another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedMap {
    name: MapName,
    source: TypedObject,
    target: TypedObject,
}

impl TypedMap {
    /// Creates a typed map between two learner-visible objects.
    pub fn new(name: MapName, source: TypedObject, target: TypedObject) -> Self {
        Self {
            name,
            source,
            target,
        }
    }

    /// Returns the map name.
    pub fn name(&self) -> &MapName {
        &self.name
    }

    /// Returns the source object.
    pub fn source(&self) -> &TypedObject {
        &self.source
    }

    /// Returns the target object.
    pub fn target(&self) -> &TypedObject {
        &self.target
    }

    /// Composes this map with the next map.
    pub fn then(&self, next: &TypedMap) -> Result<TypedMap, CategoryLensError> {
        if self.target != next.source {
            return Err(CategoryLensError::composition_mismatch(
                "TypedMap::then",
                "the first map's target object must equal the next map's source object",
            ));
        }

        Ok(TypedMap::new(
            MapName::composed(&self.name, &next.name)?,
            self.source.clone(),
            next.target.clone(),
        ))
    }
}

impl fmt::Display for TypedMap {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}: {} -> {}",
            self.name, self.source, self.target
        )
    }
}

impl<'a> Shr<&'a TypedMap> for &'a TypedMap {
    type Output = Result<TypedMap, CategoryLensError>;

    fn shr(self, right: &'a TypedMap) -> Self::Output {
        self.then(right)
    }
}

/// A checked sequence of composable maps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositionTrace {
    maps: Vec<TypedMap>,
}

impl CompositionTrace {
    /// Creates a non-empty sequence whose adjacent maps compose.
    pub fn from_maps(maps: impl IntoIterator<Item = TypedMap>) -> Result<Self, CategoryLensError> {
        let maps = maps.into_iter().collect::<Vec<_>>();
        if maps.is_empty() {
            return Err(CategoryLensError::empty_composition(
                "CompositionTrace::from_maps",
                "a trace needs at least one typed map",
            ));
        }

        for pair in maps.windows(2) {
            let previous = &pair[0];
            let next = &pair[1];
            previous.then(next)?;
        }

        Ok(Self { maps })
    }

    /// Iterates over the maps in learner order.
    pub fn maps(&self) -> impl ExactSizeIterator<Item = &TypedMap> + '_ {
        self.maps.iter()
    }

    /// Returns the single composite map represented by the trace.
    pub fn composite(&self) -> Result<TypedMap, CategoryLensError> {
        let mut maps = self.maps.iter();
        let first = maps
            .next()
            .ok_or(CategoryLensError::empty_composition(
                "CompositionTrace::composite",
                "a trace needs at least one typed map",
            ))?
            .clone();

        maps.try_fold(first, |composite, next| composite.then(next))
    }
}

#[cfg(test)]
mod tests {
    use super::{CategoryLensError, CompositionTrace, MapName, ObjectName, TypedMap, TypedObject};

    fn object(name: ObjectName) -> TypedObject {
        TypedObject::new(name)
    }

    fn map(name: MapName, source: TypedObject, target: TypedObject) -> TypedMap {
        TypedMap::new(name, source, target)
    }

    #[test]
    fn object_names_reject_empty_input() {
        let error = ObjectName::try_from("   ");
        assert!(matches!(error, Err(CategoryLensError::EmptyName { .. })));
    }

    #[test]
    fn maps_compose_when_middle_object_matches() -> Result<(), CategoryLensError> {
        let features = object(ObjectName::try_from("FeatureVector")?);
        let score = object(ObjectName::try_from("PreActivation")?);
        let prediction = object(ObjectName::try_from("Prediction")?);
        let raw_score = map(
            MapName::try_from("raw_score")?,
            features.clone(),
            score.clone(),
        );
        let sigmoid = map(MapName::try_from("sigmoid")?, score, prediction.clone());

        let composite = (&raw_score >> &sigmoid)?;

        assert_eq!(composite.source(), &features);
        assert_eq!(composite.target(), &prediction);
        assert_eq!(
            composite.to_string(),
            "sigmoid after raw_score: FeatureVector -> Prediction"
        );
        Ok(())
    }

    #[test]
    fn maps_reject_mismatched_middle_object() -> Result<(), CategoryLensError> {
        let features = object(ObjectName::try_from("FeatureVector")?);
        let prediction = object(ObjectName::try_from("Prediction")?);
        let loss = object(ObjectName::try_from("Loss")?);
        let raw_score = map(
            MapName::try_from("raw_score")?,
            features,
            object(ObjectName::try_from("PreActivation")?),
        );
        let judge = map(MapName::try_from("squared_error")?, prediction, loss);

        let error = &raw_score >> &judge;

        assert!(matches!(
            error,
            Err(CategoryLensError::CompositionMismatch { .. })
        ));
        Ok(())
    }

    #[test]
    fn composition_trace_collapses_to_one_map() -> Result<(), CategoryLensError> {
        let features = object(ObjectName::try_from("FeatureVector")?);
        let score = object(ObjectName::try_from("PreActivation")?);
        let prediction = object(ObjectName::try_from("Prediction")?);
        let loss = object(ObjectName::try_from("Loss")?);
        let trace = CompositionTrace::from_maps([
            map(
                MapName::try_from("raw_score")?,
                features.clone(),
                score.clone(),
            ),
            map(MapName::try_from("sigmoid")?, score, prediction.clone()),
            map(
                MapName::try_from("squared_error")?,
                prediction,
                loss.clone(),
            ),
        ])?;

        let composite = trace.composite()?;

        assert_eq!(trace.maps().len().to_string(), "3");
        assert_eq!(composite.source(), &features);
        assert_eq!(composite.target(), &loss);
        Ok(())
    }
}
