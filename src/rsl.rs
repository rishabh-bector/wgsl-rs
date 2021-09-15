use nom::{
    branch::alt,
    bytes::complete::{tag, take, take_while},
    combinator::{consumed, opt},
    multi::fold_many0,
    sequence::tuple,
    IResult,
};
use serde::Serialize;
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

pub fn parse(i: &str) -> SyntaxTree {
    SyntaxTree::parse(i).unwrap().1
}

pub fn test() -> String {
    let test_in = "
    use camera::CameraData;

    #[vertex]
    struct VertexInput {
        position: Vec3,
        uvs: Vec2,
    }
    
    #[transport]
    pub struct VertexOutput {
        #[position]: Vec4,
        uvs: Vec2,
    }

    // This is a commment.
    fn main(
        vert: VertexInput,
        #[uniform] camera: CameraData,
    ) -> VertexOutput {
        let screen_pos = camera.view_proj * Vec4::new(vert.position, 1.0);
    
        let mut x: i32 = 0;
        loop {
            x += 1;
            break;
        }
    
        VertexOutput::new(
            screen_pos,
            vert.uvs,
        )
    }
    
    ";

    let (todo, p_out) = SyntaxTree::parse(test_in).unwrap();
    let t0 = format!(
        "\ntest input:\n-----\n{}\n\ntest output:\n-----\n{:#?}\n",
        todo, p_out
    );

    format!("\n\nTEST: \n{}\n\n", t0)
}

pub type E<'a> = nom::error::Error<&'a str>;

#[derive(Debug)]
pub struct SyntaxTree {
    pub deps: Vec<Dependency>,
    pub nodes: Vec<Node>,
    pub exports: HashMap<String, Arc<RwLock<Node>>>,
    pub links: Vec<Arc<RwLock<Node>>>,
}

impl SyntaxTree {
    // Parse an RSL source file into an abstract syntax tree.
    pub fn parse<'a>(i: &'a str) -> IResult<&str, SyntaxTree> {
        let (rem, mut nodes) = fold_many0(Node::parse, Vec::new, |mut acc: Vec<_>, item| {
            acc.push(item);
            acc
        })(i)?;

        let (rem, _) = opt(basic::sep)(rem)?;
        if rem.len() != 0 {
            warn!("failed to parse node from:\n{}", rem);
        }

        let mut exports: HashMap<String, Node> = HashMap::new();
        for i in 0..nodes.len() {
            if nodes[i].is_pub() {
                exports.insert(nodes[i].ident().to_owned(), nodes.swap_remove(i));
            }
        }

        Ok((
            i,
            SyntaxTree {
                deps: Self::pop_deps(&mut nodes),
                exports: exports
                    .into_iter()
                    .map(|(ident, node)| (ident, Arc::new(RwLock::new(node))))
                    .collect(),
                nodes,
                links: vec![],
            },
        ))
    }

    // Drain all import statements from the tree.
    fn pop_deps(nodes: &mut Vec<Node>) -> Vec<Dependency> {
        let mut deps: Vec<Node> = vec![];
        for i in 0..nodes.len() {
            if nodes[i].is_dep() {
                deps.push(nodes.swap_remove(i))
            }
        }

        deps.into_iter()
            .map(|node| match node {
                Node::Dependency(dep) => dep,
                _ => panic!("bruh"),
            })
            .collect()
    }

    // Drain all nodes from the tree.
    pub fn pop_nodes(&mut self) -> Vec<Node> {
        self.nodes.drain(..).collect()
    }

    // Add a linked node.
    pub fn link(&mut self, node: Arc<RwLock<Node>>) {
        self.links.push(node)
    }

    // Get a node by ID
}

#[derive(Serialize, Debug)]
pub enum Node {
    Comment(String),
    Dependency(Dependency),
    Struct(Struct),
    Function(Function),
}

impl Node {
    pub fn is_dep(&self) -> bool {
        match &self {
            Node::Dependency(_) => true,
            _ => false,
        }
    }

    pub fn is_pub(&self) -> bool {
        match &self {
            Node::Struct(inner) => inner.public,
            Node::Function(inner) => inner.public,
            _ => false,
        }
    }

    pub fn ident(&self) -> &str {
        match &self {
            Node::Comment(_) => "comment",
            Node::Dependency(inner) => &inner.ident,
            Node::Struct(inner) => &inner.ident,
            Node::Function(inner) => &inner.ident,
        }
    }

    pub fn parse<'a>(i: &'a str) -> IResult<&str, Node> {
        trace!("maybe node: {:?}", i);

        let (rem, _) = basic::sep(i)?;
        let (rem, node) = alt((
            Self::parse_dependency,
            Self::parse_struct,
            Self::parse_function,
            Self::parse_comment,
        ))(rem)?;

        trace!("found node: {:#?}", node);
        Ok((rem, node))
    }

    fn parse_dependency<'a>(i: &'a str) -> IResult<&str, Node> {
        let (rem, dep) = Dependency::parse(i)?;
        Ok((rem, Node::Dependency(dep)))
    }

    fn parse_struct<'a>(i: &'a str) -> IResult<&str, Node> {
        let (rem, parsed_struct) = Struct::parse(i)?;
        Ok((rem, Node::Struct(parsed_struct)))
    }

    fn parse_function<'a>(i: &'a str) -> IResult<&str, Node> {
        let (rem, func) = Function::parse(i)?;
        Ok((rem, Node::Function(func)))
    }

    fn parse_comment<'a>(i: &'a str) -> IResult<&str, Node> {
        let (rem, comment) = basic::comment(i)?;
        Ok((rem, Node::Comment(comment.0)))
    }
}

#[derive(Serialize, Debug, PartialEq, Eq)]
pub struct Dependency {
    pub ident: String,
    pub scope: Scope,
}

#[derive(Serialize, Debug, PartialEq, Eq)]
pub enum Scope {
    Module { name: String, stage: ShaderStage },
    Common { name: String },
}

#[derive(Serialize, Debug, PartialEq, Eq)]
pub enum ShaderStage {
    Vertex,
    Fragment,
}

impl Dependency {
    pub fn parse(i: &str) -> IResult<&str, Dependency> {
        trace!("maybe dependency: {:?}", i);
        let (input, (_, _, scope, _, ident, _)) = tuple((
            basic::k_use,
            basic::sep,
            basic::ident,
            tag("::"),
            basic::ident,
            tag(";"),
        ))(i)?;

        let dep = Dependency {
            scope: match scope {
                "vert" => Scope::Module {
                    name: "vert".to_owned(),
                    stage: ShaderStage::Vertex,
                },
                "frag" => Scope::Module {
                    name: "frag".to_owned(),
                    stage: ShaderStage::Fragment,
                },
                _ => Scope::Common {
                    name: scope.to_owned(),
                },
            },
            ident: ident.to_owned(),
        };

        debug!("found dependency: {:#?}", dep);
        Ok((input, dep))
    }
}

#[derive(Serialize, Debug)]
pub struct Struct {
    pub ident: String,
    pub public: bool,
    pub entries: Vec<StructEntry>,
    pub attr: Option<Attr>,
}

impl Struct {
    pub fn parse<'a>(i: &'a str) -> IResult<&str, Struct> {
        trace!("maybe struct: {:?}", i);
        let mut attr: Option<Attr> = None;
        let mut rem: &str = i;

        if let Some((p_rem, (_, p_attr))) = consumed(Attr::parse)(i).ok() {
            match p_attr {
                Attr::Vertex => attr = Some(Attr::Vertex),
                Attr::Uniform => attr = Some(Attr::Uniform),
                Attr::Transport => attr = Some(Attr::Transport),
                _ => {}
            }
            rem = p_rem;
        }

        let (rem, _) = opt(basic::sep)(rem)?;
        let (rem, public) = opt(basic::k_pub)(rem)?;
        let public = match public {
            Some(_) => true,
            None => false,
        };

        let (rem, (_, _, _, ident, _, _, _, _)) = tuple((
            basic::sep,
            basic::k_struct,
            basic::sep,
            basic::ident,
            basic::sep,
            take_while(|c| c != '{'),
            take(1usize),
            basic::sep1,
        ))(rem)?;

        let (rem, entries) = fold_many0(StructEntry::parse, Vec::new, |mut acc: Vec<_>, item| {
            acc.push(item);
            acc
        })(rem)?;

        let (rem, _) = tuple((basic::sep, tag("}")))(rem)?;
        let parsed_struct = Struct {
            ident: ident.to_owned(),
            public,
            entries,
            attr,
        };

        debug!("found struct: {:#?}", parsed_struct);
        Ok((rem, parsed_struct))
    }
}

#[derive(Serialize, Debug)]
pub struct StructEntry {
    pub ident: String,
    pub item_type: Type,
    pub location: Option<u32>,
    pub built_in: Option<String>,
}

impl StructEntry {
    pub fn parse<'a>(i: &'a str) -> IResult<&str, StructEntry> {
        let (input, _) = opt(basic::sep)(i)?;

        if let Some((rem, (taken, attr))) = consumed(Attr::parse)(input).ok() {
            if let Attr::Builtin(p_built_in) = attr {
                let (input, (_, item_type, _)) = tuple((tag(": "), Type::parse, tag(",")))(rem)?;
                return Ok((
                    input,
                    StructEntry {
                        ident: p_built_in.to_owned(),
                        built_in: Some(p_built_in),
                        item_type,
                        location: None,
                    },
                ));
            } else {
                panic!("unknown struct item attribute: {}", taken);
            }
        }

        let (input, (ident, _, item_type, _)) =
            tuple((basic::ident, tag(": "), Type::parse, tag(",")))(input)?;

        Ok((
            input,
            StructEntry {
                ident: ident.to_owned(),
                item_type,
                location: None,
                built_in: None,
            },
        ))
    }
}

#[derive(Serialize, Debug, PartialEq, Eq)]
pub enum Attr {
    Vertex,
    Uniform,
    Transport,
    Builtin(String),
}

impl Attr {
    pub fn from_rsl(i: String) -> Self {
        match i.as_str() {
            "vertex" => Self::Vertex,
            "uniform" => Self::Uniform,
            "transport" => Self::Transport,
            other => Self::Builtin(other.to_owned()),
        }
    }

    pub fn parse<'a>(i: &'a str) -> IResult<&str, Attr> {
        let (input, (_, p_type, _)) =
            tuple((tag("#["), take_while(|c| c != ']'), take(1usize)))(i)?;
        Ok((input, Attr::from_rsl(p_type.to_owned())))
    }
}

#[derive(Serialize, Debug)]
pub struct Function {
    pub ident: String,
    pub public: bool,
    pub args: Vec<Arg>,
    pub output: Option<Type>,
    pub block: Block,
}

impl Function {
    pub fn parse<'a>(i: &'a str) -> IResult<&str, Function> {
        trace!("maybe function: {:?}", i);
        let (rem, public) = opt(basic::k_pub)(i)?;
        let public = match public {
            Some(_) => true,
            None => false,
        };

        let (rem, (_, _, _, ident, _)) =
            tuple((basic::sep, basic::k_fn, basic::sep, basic::ident, tag("(")))(rem)?;

        let (rem, args) = fold_many0(Arg::parse, Vec::new, |mut acc: Vec<_>, item| {
            acc.push(item);
            acc
        })(rem)?;

        let (rem, _) = tag(")")(rem)?;
        let (rem, output) = opt(Self::parse_output)(rem)?;
        let (rem, _) = basic::sep(rem)?;
        let (rem, block) = Block::parse(rem)?;

        let func = Function {
            ident: ident.to_owned(),
            public,
            args,
            output,
            block,
        };

        debug!("found function: {:#?}", func);
        Ok((rem, func))
    }

    fn parse_output<'a>(i: &'a str) -> IResult<&str, Type> {
        let (rem, (_, _, _, output, _)) =
            tuple((basic::sep, tag("->"), basic::sep, Type::parse, basic::sep))(i)?;
        Ok((rem, output))
    }
}

#[derive(Serialize, Debug)]
pub struct Arg {
    ident: String,
    attr: Option<Attr>,
    item_type: Type,
}

impl Arg {
    pub fn parse<'a>(i: &'a str) -> IResult<&str, Arg> {
        let mut attr: Option<Attr> = None;
        let mut rem: &str = i;
        if let Some((p_rem, (_, p_attr))) = consumed(Attr::parse)(i).ok() {
            match p_attr {
                Attr::Vertex => attr = Some(Attr::Vertex),
                Attr::Uniform => attr = Some(Attr::Uniform),
                Attr::Transport => attr = Some(Attr::Transport),
                _ => {}
            }
            rem = p_rem;
        }

        let (rem, _) = opt(basic::sep)(rem)?;
        let (rem, (ident, _, _, item_type)) =
            tuple((basic::ident, tag(":"), basic::sep, Type::parse))(rem)?;

        let (rem, _) = opt(tag(","))(rem)?;
        let (rem, _) = opt(basic::sep)(rem)?;

        Ok((
            rem,
            Arg {
                ident: ident.to_owned(),
                item_type,
                attr,
            },
        ))
    }
}

#[derive(Serialize, Debug)]
pub struct Block {
    pub body: Vec<Statement>,
    pub tail: Option<Expression>,
}

impl Block {
    pub fn parse<'a>(i: &'a str) -> IResult<&str, Block> {
        trace!("maybe block: {:?}", i);
        let (rem, _) = tuple((tag("{"), basic::sep))(i)?;
        let (rem, body) = fold_many0(Statement::parse, Vec::new, |mut acc: Vec<_>, item| {
            acc.push(item);
            acc
        })(rem)?;

        let (rem, _) = opt(basic::sep)(rem)?;
        let (rem, _maybe_end) = opt(tag("}"))(rem)?;

        let block = Block { tail: None, body };
        debug!("found block: {:#?}", block);
        return Ok((rem, block));
    }
}

#[derive(Serialize, Debug)]
pub enum Statement {
    // Item Declarations:
    //
    // let <ident>: item_type = <expr>;
    // let <ident> = <expr>;
    Decl {
        ident: String,
        item_type: Option<Type>,
        expr: Expression,
    },
    // line:
    //  <expr>;
    // void block:
    //  loop {}
    Expr {
        expr: Expression,
        output: bool,
    },
}

impl Statement {
    pub fn parse<'a>(i: &'a str) -> IResult<&str, Statement> {
        trace!("maybe statement: {:?}", i);
        let (rem, _) = opt(basic::sep)(i)?;
        let mut count: usize = 0;
        let mut state_len: usize = 0;
        for c in rem.chars() {
            match c {
                '{' => {
                    count += 1;
                }
                '}' => {
                    if count == 0 {
                        break;
                    }
                    count -= 1;
                    if count == 0 {
                        state_len += 1;
                        break;
                    }
                }
                ';' => {
                    if count == 0 {
                        state_len += 1;
                        break;
                    }
                }
                _ => {}
            }
            state_len += 1;
        }

        let (extra, rem) = take(state_len)(rem)?;
        let (extra, _) = opt(basic::sep)(extra)?;
        debug!("maybe statement_counted: {}", rem);

        // declarations
        let maybe = Self::parse_let(rem).ok();
        if let Some((_, statement)) = maybe {
            debug!("found statement decl: {:#?}", statement);
            return Ok((extra, statement));
        }

        // naked expressions
        let (_, (expr, maybe_end)) = tuple((Expression::parse, opt(tag(";"))))(rem)?;
        // if rem.len() > 0 {
        //     let rem_ = opt(basic::sep)(rem)?;
        //     rem = rem_.0;
        // }

        let statement = Statement::Expr {
            expr,
            output: maybe_end.is_none(),
        };
        debug!("found statement expr: {:#?}", statement);
        return Ok((extra, statement));
    }

    fn parse_let<'a>(i: &'a str) -> IResult<&str, Statement> {
        let (rem, (_, _, ident)) = tuple((basic::k_let, basic::sep, basic::ident))(i)?;
        let (rem, item_type) = opt(Self::parse_type)(rem)?;
        let (rem, (_, _, _, expr, _, _)) = tuple((
            basic::sep,
            tag("="),
            basic::sep,
            Expression::parse,
            tag(";"),
            basic::sep,
        ))(rem)?;

        Ok((
            rem,
            Statement::Decl {
                ident: ident.to_owned(),
                item_type,
                expr,
            },
        ))
    }

    fn parse_type<'a>(i: &'a str) -> IResult<&str, Type> {
        let (rem, (_, _, item_type, _)) =
            tuple((tag(":"), basic::sep, Type::parse, basic::sep))(i)?;
        Ok((rem, item_type))
    }
}

#[derive(Serialize, Debug)]
pub enum Iteration {
    Range { from: u32, to: u32 },
    Array { ident: String },
    Infinite,
}

#[derive(Serialize, Debug)]
pub enum Expression {
    Line(String),
    Bloc(Box<Bloc>),
}

impl Expression {
    const END_CHARS: &'static str = ";{";

    pub fn parse<'a>(i: &'a str) -> IResult<&str, Expression> {
        trace!("maybe expression: {:?}", i);
        if i.len() == 0 {
            return Err(nom::Err::Error(nom::error_position!(
                "no expression",
                nom::error::ErrorKind::Fail
            )));
        }

        let (rem, maybe_bloc) = opt(Bloc::parse)(i)?;
        if let Some(bloc) = maybe_bloc {
            debug!("found expression bloc: {:#?}", bloc);
            return Ok((rem, Expression::Bloc(Box::new(bloc))));
        }

        let (rem, outer) = take_while(|c| !Self::END_CHARS.contains(c))(i)?;
        let expr = Expression::Line(outer.to_owned());

        // if rem.len() > 0 && &rem[..1] == "{" {
        //     let (p_rem, p_block) = Block::parse(rem)?;
        //     rem = p_rem;
        //     expr.block = Some(Rc::new(p_block));
        // }

        debug!("found expression: {:#?}", expr);
        Ok((rem, expr))
    }
}

#[derive(Serialize, Debug)]
pub enum Bloc {
    If {
        cond: Expression,
        block: Box<Block>,
    },
    Loop {
        item: Option<String>,
        iter: Iteration,
        block: Box<Block>,
    },
}

impl Bloc {
    pub fn parse<'a>(i: &'a str) -> IResult<&'a str, Bloc> {
        alt((Self::parse_if, Self::parse_loop))(i)
    }

    fn parse_if<'a>(i: &'a str) -> IResult<&str, Bloc> {
        trace!("maybe bloc_if: {:?}", i);
        let (rem, (_, _, cond, _, block)) = tuple((
            basic::k_if,
            basic::sep,
            Expression::parse,
            basic::sep,
            Block::parse,
        ))(i)?;

        let bloc_if = Bloc::If {
            cond,
            block: Box::new(block),
        };

        debug!("found bloc_if: {:#?}", bloc_if);
        Ok((rem, bloc_if))
    }

    fn parse_loop<'a>(i: &'a str) -> IResult<&str, Bloc> {
        trace!("maybe bloc_loop: {:?}", i);
        let (rem, maybe_bloc) = opt(Self::parse_loop_infinite)(i)?;
        if let Some(bloc) = maybe_bloc {
            return Ok((rem, bloc));
        }
        IResult::Err(nom::Err::Error(nom::error::Error::new(
            "todo: finish loop parser",
            nom::error::ErrorKind::Fail,
        )))
    }

    fn parse_loop_infinite<'a>(i: &'a str) -> IResult<&str, Bloc> {
        let (rem, (_, _, block)) = tuple((basic::k_loop, basic::sep, Block::parse))(i)?;
        let bloc_loop = Bloc::Loop {
            item: None,
            iter: Iteration::Infinite,
            block: Box::new(block),
        };

        debug!("found bloc_loop: {:#?}", bloc_loop);
        Ok((rem, bloc_loop))
    }
}

#[derive(Serialize, Debug)]
pub enum Type {
    Bool,
    I32,
    U32,
    F32,
    Vec2,
    Vec3,
    Vec4,
    Mat2,
    Mat3,
    Mat4,
    Tex2,
    Samp,
    User(String),
}

impl Type {
    pub fn from_rsl(i: String) -> Self {
        match i.as_str() {
            "bool" => Self::Bool,
            "i32" => Self::I32,
            "u32" => Self::U32,
            "f32" => Self::F32,
            "Vec2" => Self::Vec2,
            "Vec3" => Self::Vec3,
            "Vec4" => Self::Vec4,
            "Mat2" => Self::Mat2,
            "Mat3" => Self::Mat3,
            "Mat4" => Self::Mat4,
            "Texture2" => Self::Tex2,
            "Sampler2" => Self::Samp,
            other => Self::User(i),
        }
    }

    pub fn to_wgsl(&self) -> &str {
        match self {
            Type::Bool => "bool",
            Type::I32 => "i32",
            Type::U32 => "u32",
            Type::F32 => "f32",
            Type::Vec2 => "vec2<f32>",
            Type::Vec3 => "vec3<f32>",
            Type::Vec4 => "vec4<f32>",
            Type::Mat2 => "mat2x2<f32>",
            Type::Mat3 => "mat3x3<f32>",
            Type::Mat4 => "mat4x4<f32>",
            Type::Tex2 => "texture_2d<f32>",
            Type::Samp => "sampler",
            Type::User(ident) => &ident,
        }
    }

    pub fn parse<'a>(i: &'a str) -> IResult<&str, Type> {
        let (input, p_type) = basic::ident(i)?;
        let mut p_type = p_type.to_owned();
        p_type.retain(|c| !c.is_whitespace());
        Ok((input, Type::from_rsl(p_type)))
    }
}

pub mod basic {
    use super::E;
    use nom::{
        branch::alt,
        bytes::complete::{tag, take_while, take_while1},
        character::complete::{alpha1, alphanumeric1},
        combinator::recognize,
        multi::many0,
        sequence::{pair, tuple},
        IResult,
    };

    pub fn file_name<'a>(i: &'a str) -> IResult<&'a str, &'a str, E> {
        recognize(tuple((ident, tag(".rsl"))))(i)
    }

    pub fn ident<'a>(i: &'a str) -> IResult<&'a str, &'a str, E> {
        recognize(pair(
            alt((alpha1, tag("_"))),
            many0(alt((alphanumeric1, tag("_")))),
        ))(i)
    }

    pub struct Comment(pub String);

    pub fn comment<'a>(i: &'a str) -> IResult<&'a str, Comment, E> {
        let (input, (_, comment)) = tuple((tag("//"), take_while(|c| c != '\n')))(i)?;
        Ok((input, Comment(comment.to_owned())))
    }

    const SEP_CHARS: &str = " \t\r\n";

    pub fn sep1<'a>(i: &'a str) -> IResult<&'a str, &'a str, E> {
        take_while1(move |c| SEP_CHARS.contains(c))(i)
    }

    pub fn sep<'a>(i: &'a str) -> IResult<&'a str, &'a str, E> {
        take_while(move |c| SEP_CHARS.contains(c))(i)
    }

    pub fn k_pub<'a>(i: &'a str) -> IResult<&'a str, &'a str, E> {
        tag("pub")(i)
    }

    pub fn k_use<'a>(i: &'a str) -> IResult<&'a str, &'a str, E> {
        tag("use")(i)
    }

    pub fn k_struct<'a>(i: &'a str) -> IResult<&'a str, &'a str, E> {
        tag("struct")(i)
    }

    pub fn k_fn<'a>(i: &'a str) -> IResult<&'a str, &'a str, E> {
        tag("fn")(i)
    }

    pub fn k_if<'a>(i: &'a str) -> IResult<&'a str, &'a str, E> {
        tag("if")(i)
    }

    pub fn k_let<'a>(i: &'a str) -> IResult<&'a str, &'a str, E> {
        tag("let")(i)
    }

    pub fn k_for<'a>(i: &'a str) -> IResult<&'a str, &'a str, E> {
        tag("for")(i)
    }

    pub fn k_loop<'a>(i: &'a str) -> IResult<&'a str, &'a str, E> {
        tag("loop")(i)
    }
}
