use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_LIB_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_LIB_NAME");
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_CXX");
    println!("cargo:rerun-if-changed=native/ai_llama_bridge.cpp");
    println!("cargo:rerun-if-changed=native/ai_llama_bridge.h");

    if env::var_os("CARGO_FEATURE_LLAMA_CPP").is_none() {
        return;
    }
    if env::var_os("LLAMA_CPP_SKIP_NATIVE").is_some() {
        println!("cargo:warning=llama.cpp native bridge build skipped by LLAMA_CPP_SKIP_NATIVE");
        return;
    }

    let include_dir = required_path("LLAMA_CPP_INCLUDE_DIR");
    let lib_dir = required_path("LLAMA_CPP_LIB_DIR");
    let lib_name = env::var("LLAMA_CPP_LIB_NAME").unwrap_or_else(|_| "llama".to_owned());
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    let target = env::var("TARGET").expect("TARGET is set by Cargo");
    let object = out_dir.join(if target.contains("msvc") {
        "ai_llama_bridge.obj"
    } else {
        "ai_llama_bridge.o"
    });
    let static_lib = out_dir.join(if target.contains("msvc") {
        "ai_llama_bridge.lib"
    } else {
        "libai_llama_bridge.a"
    });
    let source = Path::new("native/ai_llama_bridge.cpp");

    if target.contains("msvc") {
        compile_msvc(source, &include_dir, &object, &static_lib);
    } else {
        compile_gnu(source, &include_dir, &object, &static_lib);
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=ai_llama_bridge");
    println!("cargo:rustc-link-lib=dylib={lib_name}");
}

fn required_path(name: &str) -> PathBuf {
    let value = env::var_os(name).unwrap_or_else(|| {
        panic!(
            "{name} must point to the llama.cpp {} directory when the llama-cpp feature is enabled",
            if name.ends_with("INCLUDE_DIR") {
                "include"
            } else {
                "library"
            }
        )
    });
    let path = PathBuf::from(value);
    assert!(
        path.is_dir(),
        "{name} does not name a directory: {}",
        path.display()
    );
    path
}

fn compiler() -> OsString {
    env::var_os("LLAMA_CPP_CXX").unwrap_or_else(|| {
        if cfg!(target_env = "msvc") {
            OsString::from("cl.exe")
        } else {
            OsString::from("c++")
        }
    })
}

fn compile_msvc(source: &Path, include_dir: &Path, object: &Path, library: &Path) {
    let compiler = compiler();
    let object_arg = format!("/Fo{}", object.display());
    let status = Command::new(&compiler)
        .args(["/nologo", "/std:c++17", "/EHsc", "/MD", "/O2"])
        .arg(format!("/I{}", include_dir.display()))
        .arg("/c")
        .arg(source)
        .arg(object_arg)
        .status()
        .unwrap_or_else(|error| panic!("failed to invoke {}: {error}", compiler.to_string_lossy()));
    assert!(
        status.success(),
        "C++ bridge compilation failed with {status}"
    );

    let status = Command::new("lib.exe")
        .args(["/nologo"])
        .arg(format!("/OUT:{}", library.display()))
        .arg(object)
        .status()
        .expect("failed to invoke lib.exe; use a Visual Studio developer prompt");
    assert!(status.success(), "C++ bridge archive failed with {status}");
}

fn compile_gnu(source: &Path, include_dir: &Path, object: &Path, library: &Path) {
    let compiler = compiler();
    let status = Command::new(&compiler)
        .args(["-std=c++17", "-O3", "-fPIC", "-I"])
        .arg(include_dir)
        .args(["-c"])
        .arg(source)
        .args(["-o"])
        .arg(object)
        .status()
        .unwrap_or_else(|error| panic!("failed to invoke {}: {error}", compiler.to_string_lossy()));
    assert!(
        status.success(),
        "C++ bridge compilation failed with {status}"
    );

    let status = Command::new("ar")
        .args(["crus"])
        .arg(library)
        .arg(object)
        .status()
        .expect("failed to invoke ar");
    assert!(status.success(), "C++ bridge archive failed with {status}");
}
