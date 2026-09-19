//! Parsing what Prometheus actually sends.
//!
//! Every fixture below was captured from the lab's own server at
//! http://10.10.0.1:9090 rather than written from the documentation, including
//! the error body and its HTTP 400.

use sm_differ::measure::Refusal;
use sm_differ::prom::parse_range_response;

/// Verbatim from the lab, trimmed to two series.
const REAL_OK: &str = r#"{"status":"success","data":{"resultType":"matrix","result":[
{"metric":{"__name__":"up","arch":"sm_120","instance":"10.10.0.2:8080","job":"llama-swap","node":"laptop"},
 "values":[[1789777366,"1"],[1789777486,"1"],[1789777606,"1"],[1789777726,"1"]]},
{"metric":{"__name__":"up","arch":"sm_86","instance":"localhost:8080","job":"llama-swap","node":"desktop"},
 "values":[[1789777366,"1"],[1789777486,"1"]]}]}}"#;

/// Verbatim from the lab. This arrives with HTTP status 400, which is why the
/// client disables ureq's status-as-error -- otherwise this message is lost.
const REAL_ERR: &str = r#"{"status":"error","errorType":"bad_data","error":"invalid parameter \"query\": 1:10: parse error: unexpected identifier \"not\""}"#;

#[test]
fn parses_a_real_range_response() {
    let series = parse_range_response(REAL_OK).expect("real response should parse");
    assert_eq!(series.len(), 2);

    // The arch label the whole tool pivots on is already there; this tool does
    // not have to invent it.
    assert_eq!(series[0].arch(), Some("sm_120"));
    assert_eq!(series[0].node(), Some("laptop"));
    assert_eq!(series[1].arch(), Some("sm_86"));
    assert_eq!(series[1].node(), Some("desktop"));

    // Values arrive as JSON strings and come back as numbers.
    assert_eq!(series[0].samples.len(), 4);
    assert_eq!(series[0].samples[0], (1789777366.0, 1.0));
}

/// The error body is surfaced with its own message, not flattened to a status.
#[test]
fn a_prometheus_error_carries_its_message() {
    let err = parse_range_response(REAL_ERR).unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("bad_data"), "lost the errorType: {msg}");
    assert!(msg.contains("parse error"), "lost the diagnostic: {msg}");
}

/// NaN is a legitimate Prometheus value -- it is *why* values are strings --
/// and it must reach the refusal rather than becoming a number.
#[test]
fn nan_survives_parsing_and_is_refused_downstream() {
    let body = r#"{"status":"success","data":{"resultType":"matrix","result":[
        {"metric":{"arch":"sm_120"},"values":[[1.0,"NaN"],[2.0,"NaN"]]}]}}"#;
    let series = parse_range_response(body).expect("NaN is valid Prometheus, so it must parse");
    assert!(series[0].samples[0].1.is_nan());

    // ...and then it is refused by name, not silently zeroed.
    match series.into_iter().next().unwrap().into_sample(2) {
        Err(Refusal::NotFinite) => {}
        other => panic!("a series of NaNs must refuse, got {other:?}"),
    }
}

#[test]
fn infinities_are_refused_the_same_way() {
    let body = r#"{"status":"success","data":{"resultType":"matrix","result":[
        {"metric":{},"values":[[1.0,"+Inf"],[2.0,"1.0"]]}]}}"#;
    let series = parse_range_response(body).unwrap();
    match series.into_iter().next().unwrap().into_sample(2) {
        Err(Refusal::NotFinite) => {}
        other => panic!("an infinity must refuse, got {other:?}"),
    }
}

/// A value that is not a number at all is a protocol violation, and it is an
/// error rather than a zero. This is the line where a careless
/// `unwrap_or(0.0)` would have put the silent zero back.
#[test]
fn a_non_numeric_value_is_an_error_not_a_zero() {
    let body = r#"{"status":"success","data":{"resultType":"matrix","result":[
        {"metric":{},"values":[[1.0,"banana"]]}]}}"#;
    let err = parse_range_response(body).unwrap_err();
    assert!(
        format!("{err:#}").contains("banana"),
        "the offending value should be named: {err:#}"
    );
}

/// query_range always returns a matrix. Anything else means the caller reached
/// the wrong endpoint, and guessing the shape turns a clear mistake into a
/// confusing one.
#[test]
fn a_non_matrix_result_is_rejected() {
    let body = r#"{"status":"success","data":{"resultType":"vector","result":[]}}"#;
    let err = parse_range_response(body).unwrap_err();
    assert!(format!("{err:#}").contains("matrix"), "{err:#}");
}

/// No series is a legitimate answer -- the metric may simply not have been
/// scraped in that window. It is an empty result, not an error and not a zero.
#[test]
fn an_empty_result_is_empty_not_zero() {
    let body = r#"{"status":"success","data":{"resultType":"matrix","result":[]}}"#;
    assert!(parse_range_response(body).unwrap().is_empty());
}

/// A series with a single reading cannot support a verdict, and says so with
/// the refusal that names the reason rather than inventing a target count.
#[test]
fn a_single_scrape_cannot_support_a_verdict() {
    let body = r#"{"status":"success","data":{"resultType":"matrix","result":[
        {"metric":{},"values":[[1.0,"175.9"]]}]}}"#;
    let series = parse_range_response(body).unwrap();
    match series.into_iter().next().unwrap().into_sample(1) {
        Err(Refusal::NoVariance { got: 1 }) => {}
        other => panic!("expected NoVariance, got {other:?}"),
    }
}

/// Garbage in, error out -- never a default-constructed anything.
#[test]
fn malformed_json_is_an_error() {
    for body in ["", "not json", "{}", r#"{"status":"weird"}"#] {
        assert!(
            parse_range_response(body).is_err(),
            "{body:?} should not parse"
        );
    }
}
