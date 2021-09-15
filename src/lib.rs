extern crate pretty_env_logger;
extern crate proc_macro;
#[macro_use]
extern crate log;

use anyhow::{anyhow, Result};
use proc_macro::TokenStream;
use quote::quote;
use rsl::{Dependency, Node, Scope, SyntaxTree};
use serde::Serialize;
use std::{
    collections::HashMap,
    env,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
};
use syn;

use crate::rsl::ShaderStage;

mod rsl;

#[proc_macro_attribute]
pub fn shaders(attr: TokenStream, item: TokenStream) -> TokenStream {
    std::env::set_var("RUST_LOG", "trace");
    pretty_env_logger::init();

    // let out = rsl::test();
    // return quote!(
    //     pub fn test_macro() {
    //         let out = #out;
    //         println!("{}", out);
    //     }
    // )
    // .into();

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

    let mut rsl_path = attr.to_string().replace("\"", "");
    rsl_path.retain(|c| !c.is_whitespace());

    let crate_tree = parse_rsl(&rsl_path, mod_names).unwrap();
    // let dbg = serde_json::to_string(&crate_tree).unwrap();

    let path = Path::new("crate_tree.json");
    let display = path.display();
    let mut file = match File::create(&path) {
        Err(why) => panic!("couldn't create {}: {}", display, why),
        Ok(file) => file,
    };
    match file.write_all(dbg.as_bytes()) {
        Err(why) => panic!("couldn't write to {}: {}", display, why),
        Ok(_) => println!("successfully wrote to {}", display),
    }

    quote!(
        pub fn test_macro() {
            // println!("{}", #dbg);
        }
    )
    .into()
}

#[derive(Debug)]
struct CrateTree {
    pub modules: HashMap<String, ModuleTree>,
    pub common: HashMap<String, SyntaxTree>,
}

impl CrateTree {
    pub fn link(mut self) {
        for (name, mut module) in self.modules {
            // merged module dependencies
            let mut vert_deps = module.vert.pop_deps();
            let mut frag_deps = module.frag.pop_deps();
            for i in 0..frag_deps.len() {
                if vert_deps.contains(&frag_deps[i]) {
                    frag_deps.swap_remove(i);
                }
            }
        }
    }

    pub fn clone_dep(&self, dep: &Dependency) -> Result<Arc<RwLock<Node>>> {
        Ok(match &dep.scope {
            Scope::Common { name } => self
                .common
                .get(name)
                .ok_or_else(|| anyhow!("couldn't find common module: {}", name))?
                .exports
                .get(&dep.ident)
                .ok_or_else(|| {
                    anyhow!("couldn't find dependency in module {}: {}", name, dep.ident)
                })?
                .to_owned(),
            Scope::Module { name, stage } => {
                let module = self
                    .modules
                    .get(name)
                    .ok_or_else(|| anyhow!("couldn't find shader module: {}", name))?;

                Arc::clone(
                    match stage {
                        ShaderStage::Vertex => &module.vert,
                        ShaderStage::Fragment => &module.frag,
                    }
                    .exports
                    .get(&dep.ident)
                    .ok_or_else(|| {
                        anyhow!("couldn't find dependency in module {}: {}", name, dep.ident)
                    })?,
                )
            }
        })
    }
}

#[derive(Debug)]
struct ModuleTree {
    pub vert: SyntaxTree,
    pub frag: SyntaxTree,
}

// #[derive(Serialize, Debug)]
// struct MergedTree {
//     pub vert_deps: Vec<Dependency>,
//     pub frag_deps: Vec<Dependency>,
//     pub vert: SyntaxTree,
//     pub frag: SyntaxTree,
// }

impl ModuleTree {
    pub fn link(&mut self, vert_deps: Vec<Node>, frag_deps: Vec<Node>) {}

    pub fn merge(self) -> SyntaxTree {
        let mut nodes = self.vert.nodes;
        let mut exports = self.vert.exports;
        let mut links = self.vert.links;

        nodes.extend(self.frag.nodes);
        exports.extend(self.frag.exports);
        links.extend(self.frag.links);

        SyntaxTree {
            nodes,
            exports,
            links,
        }
    }
}

fn parse_rsl(path: &str, module_names: Vec<String>) -> Result<CrateTree> {
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

    let modules: HashMap<String, ModuleTree> = module_names
        .into_iter()
        .map(|mod_name| {
            let vert = fs::read_to_string(modules_path.join(format!("{}/vert.rsl", mod_name)))
                .expect(&format!("failed to read vert.rsl in module {}", &mod_name));
            let frag = fs::read_to_string(modules_path.join(format!("{}/frag.rsl", mod_name)))
                .expect(&format!("failed to read frag.rsl in module {}", &mod_name));

            (
                mod_name,
                ModuleTree {
                    vert: rsl::parse(&vert),
                    frag: rsl::parse(&frag),
                },
            )
        })
        .collect();

    let common: HashMap<String, SyntaxTree> = fs::read_dir(common_path.clone())
        .expect(&format!(
            "unable to open common/: {}",
            common_path.to_str().unwrap()
        ))
        .map(|entry| entry.unwrap())
        .map(|entry| {
            let file_name = entry.file_name().to_str().unwrap().to_string();
            let source = fs::read_to_string(entry.path())
                .expect(&format!("failed to read common shader {}", &file_name));
            (
                rsl::basic::file_name(&file_name).unwrap().1.to_owned(),
                rsl::parse(&source),
            )
        })
        .collect();

    Ok(CrateTree { modules, common })
}
