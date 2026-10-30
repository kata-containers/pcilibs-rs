// Copyright (c) NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

//! The two ends of the fabric view's Publish stream, generated from the
//! proto, and the token both agree on.  kata-device-plugin serves the
//! stream and hands the token to the one container it gives the fabric to
//! (see [`super::ENDPOINT_ENV`] and [`super::TOKEN_ENV`]); a Fabric Manager
//! client in that container publishes with it.  A publisher on the host
//! itself reaches the plugin over [`super::HOST_SOCKET`], where filesystem
//! permissions are the credential and no token is checked.

use std::sync::Arc;

use tonic::metadata::errors::InvalidMetadataValue;
use tonic::{Request, Status};

pub use super::v1::fabric_view_service_client::FabricViewServiceClient;
pub use super::v1::fabric_view_service_server::{FabricViewService, FabricViewServiceServer};
use super::TOKEN_METADATA_KEY;

/// Attach the plugin's token to a request the way [`require_token`] reads
/// it.  Fails only for a token that is not valid metadata, which the
/// plugin never issues.
pub fn with_token<T>(
    mut request: Request<T>,
    token: &str,
) -> Result<Request<T>, InvalidMetadataValue> {
    request
        .metadata_mut()
        .insert(TOKEN_METADATA_KEY, token.parse()?);
    Ok(request)
}

/// A server interceptor admitting only requests that carry `expected`:
/// `FabricViewServiceServer::with_interceptor(service, require_token(token))`.
/// The comparison runs over every byte, so a wrong token is rejected
/// after the same number of operations whatever its first wrong byte.
pub fn require_token(
    expected: &str,
) -> impl Fn(Request<()>) -> Result<Request<()>, Status> + Clone {
    let expected: Arc<str> = Arc::from(expected);
    move |request: Request<()>| {
        let presented = request
            .metadata()
            .get(TOKEN_METADATA_KEY)
            .and_then(|value| value.to_str().ok());
        match presented {
            Some(token) if same(token.as_bytes(), expected.as_bytes()) => Ok(request),
            _ => Err(Status::unauthenticated(
                "fabric view token missing or wrong",
            )),
        }
    }
}

/// Equality without an early exit on the first differing byte.
fn same(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use tonic::Code;

    #[test]
    fn the_issued_token_is_admitted() {
        let admit = require_token("s3cret");
        let request = with_token(Request::new(()), "s3cret").unwrap();
        assert!(admit(request).is_ok());
    }

    #[test]
    fn a_wrong_missing_or_malformed_token_is_unauthenticated() {
        let admit = require_token("s3cret");
        for token in ["", "s3cres", "s3cret ", "S3CRET", "s3cret-and-more"] {
            let status = admit(with_token(Request::new(()), token).unwrap()).unwrap_err();
            assert_eq!(status.code(), Code::Unauthenticated, "{token:?}");
        }
        let status = admit(Request::new(())).unwrap_err();
        assert_eq!(status.code(), Code::Unauthenticated);
        // Not valid metadata: a token the plugin would never issue.
        assert!(with_token(Request::new(()), "line\nbreak").is_err());
    }

    #[test]
    fn same_compares_whole_slices() {
        assert!(same(b"", b""));
        assert!(same(b"abc", b"abc"));
        assert!(!same(b"abc", b"abd"));
        assert!(!same(b"abc", b"ab"));
        assert!(!same(b"ab", b"abc"));
    }

    #[test]
    fn the_interceptor_is_clonable_for_a_server() {
        let admit = require_token("t");
        let twin = admit.clone();
        assert!(twin(with_token(Request::new(()), "t").unwrap()).is_ok());
        // The generated server takes it as an interceptor.
        let _server = FabricViewServiceServer::with_interceptor(Unused, admit);
    }

    struct Unused;

    #[tonic::async_trait]
    impl FabricViewService for Unused {
        async fn publish(
            &self,
            _: Request<tonic::Streaming<super::super::v1::FabricView>>,
        ) -> Result<tonic::Response<super::super::v1::PublishSummary>, Status> {
            unreachable!()
        }
    }
}
