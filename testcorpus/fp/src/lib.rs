#[no_mangle]
pub extern "C" fn fp_arith(a: f64, b: f64) -> f64 {
    a * b + a
}

#[no_mangle]
pub extern "C" fn fp_sqrt(a: f64, b: f64) -> f64 {
    (a * b + a).sqrt()
}

#[no_mangle]
pub extern "C" fn fp_cvt(a: f64, n: i32) -> f64 {
    a * (n as f64) - 1.5
}

#[no_mangle]
pub extern "C" fn fp_f32(a: f32, b: f32) -> f32 {
    (a - b) * a / (b + 2.0)
}

#[no_mangle]
pub extern "C" fn fp_powf(a: f64, b: f64) -> f64 {
    a.powf(b) + 1.0
}

#[no_mangle]
pub extern "C" fn fp_trunc(a: f64) -> i32 {
    (a * 2.0) as i32
}

#[no_mangle]
pub extern "C" fn fp_neg(a: f64, b: f64) -> f64 {
    -(a - b).abs()
}

#[link(name = "api-ms-win-crt-math-l1-1-0", kind = "raw-dylib")]
extern "C" {
    fn pow(x: f64, y: f64) -> f64;
}

#[no_mangle]
pub extern "C" fn fp_pow_iat(a: f64, b: f64) -> f64 {
    unsafe { pow(a, b) + 1.0 }
}
