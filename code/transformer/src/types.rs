//! Domain newtypes for the tiny Transformer crate.

use std::fmt;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, Mul, Sub};

/// A model scalar used for weights, activations, scores, and normalized values.
#[derive(Debug, Clone, Copy, Default, PartialEq, PartialOrd)]
pub struct Scalar(f32);

impl Scalar {
    /// The additive identity.
    pub const ZERO: Self = Self(0.0);
    /// The multiplicative identity.
    pub const ONE: Self = Self(1.0);
    /// The most negative finite-like sentinel used in stable softmax.
    pub const NEG_INFINITY: Self = Self(f32::NEG_INFINITY);

    /// Creates a scalar from a primitive boundary value.
    pub const fn new(value: f32) -> Self {
        Self(value)
    }

    /// Returns the wrapped primitive value.
    pub const fn into_inner(self) -> f32 {
        self.0
    }

    /// Returns the absolute value.
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    /// Returns the exponential.
    pub fn exp(self) -> Self {
        Self(self.0.exp())
    }

    /// Returns the square root.
    pub fn sqrt(self) -> Self {
        Self(self.0.sqrt())
    }

    /// Returns the larger of two scalars.
    pub fn max(self, other: Self) -> Self {
        Self(self.0.max(other.0))
    }

    /// Returns true when the wrapped value is finite.
    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }
}

impl From<f32> for Scalar {
    fn from(value: f32) -> Self {
        Self(value)
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Add for Scalar {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign for Scalar {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl Sub for Scalar {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl Mul for Scalar {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl Div for Scalar {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}

impl Sum for Scalar {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, value| acc + value)
    }
}

macro_rules! usize_newtype {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(usize);

        impl $name {
            #[doc = "Creates the newtype from a primitive boundary value."]
            pub const fn new(value: usize) -> Self {
                Self(value)
            }

            #[doc = "Returns the wrapped primitive value."]
            pub const fn get(self) -> usize {
                self.0
            }
        }

        impl From<usize> for $name {
            fn from(value: usize) -> Self {
                Self(value)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }
    };
}

usize_newtype!(Dimension, "The shared size of a model vector.");
usize_newtype!(RowCount, "The number of rows in a matrix.");
usize_newtype!(ColumnCount, "The number of columns in a matrix.");
usize_newtype!(TokenCount, "The number of tokens in a sequence.");
usize_newtype!(RowIndex, "A row index into a matrix.");
usize_newtype!(ColumnIndex, "A column index into a matrix.");
