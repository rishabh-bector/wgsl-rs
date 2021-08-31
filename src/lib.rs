extern crate proc_macro;

use anyhow::{anyhow, Result};
use proc_macro::TokenStream;
use quote::quote;
use std::{collections::HashMap, env, fs, path::PathBuf};
use syn;

#[proc_macro_attribute]
pub fn shaders(attr: TokenStream, item: TokenStream) -> TokenStream {
    let ast: syn::DeriveInput = syn::parse(item).unwrap();

    let mod_enum = match ast.data {
        syn::Data::Enum(data) => Some(data),
        _ => None,
    }
    .expect("the #[shader()] macro must be an outer enum attribute");

    let mod_names: Vec<String> = mod_enum
        .variants
        .iter()
        .map(|var| var.ident.to_string().to_ascii_lowercase())
        .collect();
    let dbg = format!("{:?}", mod_names);

    let mut rsl_path = attr.to_string();
    rsl_path.retain(|c| !c.is_whitespace());

    let rsl = read_rsl(&rsl_path, mod_names).unwrap();
    quote!(
        pub fn test_macro() {
            println!(#dbg);
        }
    )
    .into()
}

struct RSL {
    modules: HashMap<String, ShaderModule>,
    common: HashMap<String, ShaderCommon>,
}

struct ShaderModule {
    vert: String,
    frag: String,
}

struct ShaderCommon(String);

fn read_rsl(path: &str, module_names: Vec<String>) -> Result<RSL> {
    let wd = env::current_dir().expect(&format!("unable to read working directory"));
    let rsl_path = wd.join(path);
    let dir = fs::read_dir(rsl_path.clone()).expect(&format!(
        "unable to open rsl directory: {}, relative to: {}",
        rsl_path.to_str().unwrap(),
        wd.to_str().unwrap()
    ));

    let paths: Vec<fs::DirEntry> = dir.map(|entry| entry.unwrap()).collect();
    if paths.len() != 2 {
        return Err(anyhow!(
            "rsl directory should contain only modules/ and common/"
        ));
    }

    let modules_path = paths
        .iter()
        .find(|path| path.file_name().to_ascii_lowercase().eq("modules"))
        .expect(&format!("couldn't find modules/ in {}", path))
        .path();
    let common_path = paths
        .iter()
        .find(|path| path.file_name().to_ascii_lowercase().eq("common"))
        .expect(&format!("couldn't find common/ in {}", path))
        .path();

    let modules: HashMap<String, ShaderModule> = module_names
        .into_iter()
        .map(|mod_name| {
            let vert = fs::read_to_string(modules_path.join(format!("{}/vert.rsl", mod_name)))
                .expect(&format!("failed to read vert.rsl in module {}", &mod_name));
            let frag = fs::read_to_string(modules_path.join(format!("{}/frag.rsl", mod_name)))
                .expect(&format!("failed to read frag.rsl in module {}", &mod_name));
            (mod_name, ShaderModule { vert, frag })
        })
        .collect();

    let common: HashMap<String, ShaderCommon> = fs::read_dir(common_path.clone())
        .expect(&format!(
            "unable to open common/: {}",
            common_path.to_str().unwrap()
        ))
        .map(|entry| entry.unwrap())
        .map(|entry| {
            let file_name = entry.file_name().to_str().unwrap().to_string();
            let source = fs::read_to_string(entry.path())
                .expect(&format!("failed to read common shader {}", &file_name));
            (file_name, ShaderCommon(source))
        })
        .collect();

    Ok(RSL { modules, common })
}
