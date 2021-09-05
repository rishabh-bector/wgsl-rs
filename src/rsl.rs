use std::rc::Rc;

use nom::{
    branch::alt,
    bytes::complete::{tag, take, take_while, take_while1},
    combinator::{consumed, map_parser, map_res, opt, recognize},
    error::{FromExternalError, ParseError},
    multi::fold_many0,
    sequence::tuple,
    IResult,
};

// Items (top level):
//  - V1 Types:
//      - Convert all types from RSL -> WGSL
//  - V1 Imports:
//      - Linking
//  - V1`Structs
//      - #[vertex] generate Rust, [[layout]]
//      - #[uniform] generate Rust, [[block]]
//      - #[transport] generate builtin #[position] binding
//      - generate constructors with ::new()
//  - V0 Functions
//      - entrypoints take uniforms as arguments
//      - functions can be split over whitespace without issue
//  - V0 Bindings
//      - group binding are generated based on func arg order
//  - V0 Global vars
//      - get rid of these?
//  - V1 Comments

pub fn parse(i: &str) -> SyntaxTree {
    todo!()
}

pub fn test() -> String {
    let test_in = "let screen_pos = camera.view_proj * Vec4::new(vert.position, 1.0);";

    type E<'a> = nom::error::Error<&'a str>;

    let (todo, p_out) = Statement::parse::<E>(test_in).unwrap();
    let t0 = format!("todo: {:?}\ndone: {:#?}", todo, p_out);

    format!("\n\nTEST: \n{}\n\n", t0)
}

#[derive(Debug)]
pub struct Function {
    pub ident: String,
    pub public: bool,
    pub args: Vec<Arg>,
    pub output: Option<Type>,
    pub block: Block,
}

impl Function {
    pub fn parse<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&str, Function> {
        let (rem, public) = opt(basic::k_pub)(i)?;
        let public = match public {
            Some(_) => true,
            None => false,
        };

        let (rem, (_, _, _, ident, _)) =
            tuple((basic::sep, basic::k_fn, basic::sep, basic::ident, tag("(")))(rem)?;

        let (rem, args) = fold_many0(Arg::parse::<E>, Vec::new, |mut acc: Vec<_>, item| {
            acc.push(item);
            acc
        })(rem)?;

        let (rem, _) = tag(")")(rem)?;
        let (rem, output) = opt(Self::parse_output::<E>)(rem)?;
        let (rem, _) = basic::sep(rem)?;
        let (rem, block) = Block::parse::<E>(rem)?;

        Ok((
            rem,
            Function {
                ident: ident.to_owned(),
                public,
                args,
                output,
                block,
            },
        ))
    }

    fn parse_output<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&str, Type> {
        let (rem, (_, _, _, output, _)) = tuple((
            basic::sep,
            tag("->"),
            basic::sep,
            Type::parse::<E>,
            basic::sep,
        ))(i)?;
        Ok((rem, output))
    }
}

#[derive(Debug)]
pub struct Arg {
    ident: String,
    attr: Option<Attr>,
    item_type: Type,
}

impl Arg {
    pub fn parse<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&str, Arg> {
        let mut attr: Option<Attr> = None;
        let mut rem: &str = i;
        if let Some((p_rem, (_, p_attr))) = consumed(Attr::parse::<E>)(i).ok() {
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
            tuple((basic::ident, tag(":"), basic::sep, Type::parse::<E>))(rem)?;

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

#[derive(Debug)]
pub struct BodyEntry {}
impl BodyEntry {
    pub fn parse<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&str, BodyEntry> {
        todo!()
    }
}

#[derive(Debug)]
pub struct Block {
    pub body: Vec<Statement>,
    pub tail: Expression,
    pub tail_statement: bool,
}

impl Block {
    pub fn parse<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&str, Block> {
        todo!()
    }
}

#[derive(Debug)]
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
    // Naked expressions:
    //
    // <expr>;
    Expr {
        expr: Expression,
    },
    //  Control blocs:
    //
    // <ident> <outer> { <block> }
    Bloc(Bloc),
}

impl Statement {
    pub fn parse<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&str, Statement> {
        // declarations & blocs
        let maybe = alt((Self::parse_let::<E>, Self::parse_bloc::<E>))(i).ok();
        if let Some((rem, statement)) = maybe {
            return Ok((rem, statement));
        }

        // naked expressions
        let (rem, expr) = Expression::parse::<E>(i)?;
        return Ok((rem, Statement::Expr { expr }));
    }

    fn parse_bloc<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&str, Statement> {
        let (rem, bloc) = Bloc::parse::<E>(i)?;
        Ok((rem, Statement::Bloc(bloc)))
    }

    fn parse_let<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&str, Statement> {
        let (rem, (_, ident)) = tuple((basic::sep, basic::ident))(i)?;
        let (rem, item_type) = opt(Self::parse_type::<E>)(rem)?;
        let (rem, (_, _, _, expr, _)) = tuple((
            basic::sep,
            tag("="),
            basic::sep,
            Expression::parse::<E>,
            tag(";"),
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

    fn parse_type<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&str, Type> {
        let (rem, (_, _, item_type, _)) =
            tuple((tag(":"), basic::sep, Type::parse::<E>, basic::sep))(i)?;
        Ok((rem, item_type))
    }
}

#[derive(Debug)]
pub enum Bloc {
    If {
        cond: Expression,
        block: Block,
    },
    Loop {
        item: Option<String>,
        iter: Iteration,
        block: Block,
    },
}

impl Bloc {
    pub fn parse<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&'a str, Bloc> {
        alt((Self::parse_if::<E>, Self::parse_loop::<E>))(i)
    }

    fn parse_if<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&str, Bloc> {
        let (rem, (_, _, cond, _, block)) = tuple((
            basic::k_if,
            basic::sep,
            Expression::parse::<E>,
            basic::sep,
            Block::parse::<E>,
        ))(i)?;

        Ok((rem, Bloc::If { cond, block }))
    }

    fn parse_loop<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&str, Bloc> {
        let (rem, maybe_bloc) = opt(Self::parse_loop_infinite::<E>)(i)?;
        if let Some(bloc) = maybe_bloc {
            return Ok((rem, bloc));
        }
        todo!()
    }

    fn parse_loop_infinite<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&str, Bloc> {
        let (rem, (_, _, block)) = tuple((basic::k_loop, basic::sep, Block::parse::<E>))(i)?;
        Ok((
            rem,
            Bloc::Loop {
                item: None,
                iter: Iteration::Infinite,
                block,
            },
        ))
    }
}

#[derive(Debug)]
pub enum Iteration {
    Range { from: u32, to: u32 },
    Array { ident: String },
    Infinite,
}

#[derive(Debug)]
pub struct Expression {
    pub outer: String,
    pub block: Option<Rc<Block>>,
}

impl Expression {
    const END_CHARS: &'static str = ";{";

    pub fn parse<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&str, Expression> {
        let (mut rem, outer) = take_while(|c| !Self::END_CHARS.contains(c))(i)?;
        let mut block: Option<Rc<Block>> = None;
        if &rem[..1] == "{" {
            let (p_rem, p_block) = Block::parse::<E>(rem)?;
            rem = p_rem;
            block = Some(Rc::new(p_block));
        }
        Ok((
            rem,
            Expression {
                outer: outer.to_owned(),
                block,
            },
        ))
    }
}

pub struct SyntaxTree {}

// Source represents a single RSL file
pub struct Source {
    pub deps: Vec<Dependency>,
}

#[derive(Debug)]
pub enum ShaderStage {
    Vertex,
    Fragment,
}

#[derive(Debug)]
pub enum Scope {
    Module { name: String, stage: ShaderStage },
    Common { name: String },
}

#[derive(Debug)]
pub struct Dependency {
    pub ident: String,
    pub scope: Scope,
}

impl Dependency {
    pub fn parse(i: &str) -> IResult<&str, Dependency> {
        let (input, (_, _, scope, _, ident, _)) = tuple((
            basic::k_use,
            basic::sep,
            basic::ident,
            tag("::"),
            basic::ident,
            tag(";"),
        ))(i)?;

        Ok((
            input,
            Dependency {
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
            },
        ))
    }
}

#[derive(Debug)]
pub struct Struct {
    pub ident: String,
    pub public: bool,
    pub entries: Vec<StructEntry>,
    pub attr: Option<Attr>,
}

impl Struct {
    pub fn parse<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&str, Struct> {
        let mut attr: Option<Attr> = None;
        let mut rem: &str = i;
        if let Some((p_rem, (_, p_attr))) = consumed(Attr::parse::<E>)(i).ok() {
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

        let (rem, entries) = fold_many0(
            StructEntry::parse::<E>,
            Vec::new,
            |mut acc: Vec<_>, item| {
                acc.push(item);
                acc
            },
        )(rem)?;

        let (rem, _) = tuple((basic::sep, tag("}")))(rem)?;

        Ok((
            rem,
            Struct {
                ident: ident.to_owned(),
                public,
                entries,
                attr,
            },
        ))
    }
}

#[derive(Debug)]
pub struct StructEntry {
    pub ident: String,
    pub item_type: Type,
    pub location: Option<u32>,
    pub built_in: Option<String>,
}

impl StructEntry {
    pub fn parse<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&str, StructEntry> {
        let (input, _) = opt(basic::sep)(i)?;

        if let Some((rem, (taken, attr))) = consumed(Attr::parse::<E>)(input).ok() {
            if let Attr::Builtin(p_built_in) = attr {
                let (input, (_, item_type, _)) =
                    tuple((tag(": "), Type::parse::<E>, tag(",")))(rem)?;
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
            tuple((basic::ident, tag(": "), Type::parse::<E>, tag(",")))(input)?;

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

#[derive(Debug, PartialEq, Eq)]
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

    pub fn parse<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&str, Attr> {
        let (input, (_, p_type, _)) =
            tuple((tag("#["), take_while(|c| c != ']'), take(1usize)))(i)?;
        Ok((input, Attr::from_rsl(p_type.to_owned())))
    }
}

#[derive(Debug)]
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

    pub fn parse<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&str, Type> {
        let (input, p_type) = basic::ident(i)?;
        let mut p_type = p_type.to_owned();
        p_type.retain(|c| !c.is_whitespace());
        Ok((input, Type::from_rsl(p_type)))
    }
}

mod basic {
    use nom::{
        branch::alt,
        bytes::complete::{tag, take_while, take_while1},
        character::complete::{alpha1, alphanumeric1},
        combinator::recognize,
        error::ParseError,
        multi::many0,
        sequence::{pair, tuple},
        IResult,
    };

    pub fn ident<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&'a str, &'a str, E> {
        recognize(pair(
            alt((alpha1, tag("_"))),
            many0(alt((alphanumeric1, tag("_")))),
        ))(i)
    }

    pub fn comment<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&'a str, &'a str, E> {
        let (input, (_, comment)) = tuple((tag("//"), take_while(|c| c != '\n')))(i)?;
        Ok((input, comment))
    }

    const SEP_CHARS: &str = " \t\r\n";

    pub fn sep1<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&'a str, &'a str, E> {
        take_while1(move |c| SEP_CHARS.contains(c))(i)
    }

    pub fn sep<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&'a str, &'a str, E> {
        take_while(move |c| SEP_CHARS.contains(c))(i)
    }

    pub fn k_pub<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&'a str, &'a str, E> {
        tag("pub")(i)
    }

    pub fn k_use<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&'a str, &'a str, E> {
        tag("use")(i)
    }

    pub fn k_struct<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&'a str, &'a str, E> {
        tag("struct")(i)
    }

    pub fn k_fn<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&'a str, &'a str, E> {
        tag("fn")(i)
    }

    pub fn k_if<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&'a str, &'a str, E> {
        tag("if")(i)
    }

    pub fn k_let<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&'a str, &'a str, E> {
        tag("let")(i)
    }

    pub fn k_for<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&'a str, &'a str, E> {
        tag("for")(i)
    }

    pub fn k_loop<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&'a str, &'a str, E> {
        tag("loop")(i)
    }
}
