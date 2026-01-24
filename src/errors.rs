// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ClustorError {
    #[error("invalid argument: {0}")]
    InvalidArg(String),
}

pub type ClustorResult<T> = Result<T, ClustorError>;
