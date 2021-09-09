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

//
//
// -----------------------------------------
// -----   RSL Compiler Architecture   -----
// -----------------------------------------
//
//
// Stage 1: AST Parser
//
//  a source is a single .rsl file.
//  each source is parsed into an abstract syntax tree.
//
//  AST Component       Structure
//  -----------------------------------------------------
//  - syntaxtree        [sourceentry]
//  - node              || comment, import, struct, function ||
//
//  - type              || all supported types ||
//  - import            { ident, scope }
//  - scope             { ident, || module{ident}, common || }
//  - struct            { ident, public, [entry], attribute? }
//  - entry             { ident, type, location?, builtin? }
//  - attribute         || vertex, transport, uniform, custom{ident} ||
//  - function          { ident, public, [arg], output, block }
//  - arg               { ident, type, attribute? }
//  - statement         || declaration, bloc, expression ||
//  - declaration       { ident, type, expression }
//  - bloc              || if, loop ||
//  - if                { expression, block }
//  - loop              { ident, iteration, block }
//  - iteration         || TODO range{u32, u32}, TODO array{ident}, infinite ||
//  - block             { [statement], expression? }
//  - expression        { string }
//  - comment           { string }
//
//
// Stage 2: Linker
//
//  vert/frag asts in each module are merged into one ast per module.
//  imports are cloned into each module ast.
//
//
// Stage 3: Rust Generator
//
//  structs with the vertex attribute are used to generate wgsl rust.
//  structs with the uniform attribute are used to generate wgsl rust.
//  the rsl module trees are now ready to be transpiled into wgsl.
//
//
// Stage 4: WGSL Transpiler
//
//  AST Component       Mode
//  -----------------------------------------------------
//  - syntaxtree        inherited
//  - node              inherited
//
//  - type              mapped
//  - import            n/a
//  - scope             n/a
//  - struct            computed (mut: transport layouts, builtin attrs, [[block]] uniforms)
//  - entry             inherited
//  - attribute         n/a
//  - function          computed (mut: uniform args)
//  - arg               inherited
//  - statement         inherited
//  - declaration       inherited
//  - bloc              inherited
//  - if                inherited
//  - loop              inherited
//  - iteration         computed (mut: todo)
//  - block             inherited
//  - expression        computed (mut: struct constructors)
//  - comment           inlined
//
//

pub fn parse(i: &str) -> SyntaxTree {
    todo!()
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

    type E<'a> = nom::error::Error<&'a str>;

    let (todo, p_out) = SyntaxTree::parse::<E>(test_in).unwrap();
    let t0 = format!(
        "\ntest input:\n-----\n{}\n\ntest output:\n-----\n{:#?}\n",
        todo, p_out
    );

    format!("\n\nTEST: \n{}\n\n", t0)
}

#[derive(Debug)]
pub struct SyntaxTree {
    pub nodes: Vec<Node>,
}

impl SyntaxTree {
    pub fn parse<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&str, SyntaxTree> {
        let (rem, nodes) = fold_many0(Node::parse::<E>, Vec::new, |mut acc: Vec<_>, item| {
            acc.push(item);
            acc
        })(i)?;

        let (rem, _) = opt(basic::sep)(rem)?;
        if rem.len() != 0 {
            warn!("failed to parse node from:\n{}", rem);
        }

        Ok((i, SyntaxTree { nodes }))
    }
}

#[derive(Debug)]
pub enum Node {
    Comment,
    Dependency(Dependency),
    Struct(Struct),
    Function(Function),
}

impl Node {
    pub fn parse<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&str, Node> {
        trace!("maybe node: {:?}", i);

        let (rem, _) = basic::sep(i)?;
        let (rem, node) = alt((
            Self::parse_dependency::<E>,
            Self::parse_struct::<E>,
            Self::parse_function::<E>,
        ))(rem)?;

        trace!("found node: {:#?}", node);
        Ok((rem, node))
    }

    fn parse_dependency<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&str, Node> {
        let (rem, dep) = Dependency::parse(i)?;
        Ok((rem, Node::Dependency(dep)))
    }

    fn parse_struct<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&str, Node> {
        let (rem, parsed_struct) = Struct::parse::<E>(i)?;
        Ok((rem, Node::Struct(parsed_struct)))
    }

    fn parse_function<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&str, Node> {
        let (rem, func) = Function::parse::<E>(i)?;
        Ok((rem, Node::Function(func)))
    }
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
        trace!("maybe function: {:?}", i);
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
pub struct Block {
    pub body: Vec<Statement>,
    pub tail: Option<Expression>,
}

impl Block {
    pub fn parse<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&str, Block> {
        trace!("maybe block: {:?}", i);
        let (rem, _) = tuple((tag("{"), basic::sep))(i)?;
        let (rem, mut body) =
            fold_many0(Statement::parse::<E>, Vec::new, |mut acc: Vec<_>, item| {
                acc.push(item);
                acc
            })(rem)?;

        let (rem, _) = opt(basic::sep)(rem)?;
        let (rem, maybe_end) = opt(tag("}"))(rem)?;

        // no tail
        if maybe_end.is_some() {
            let block = Block { tail: None, body };
            debug!("found block: {:#?}", block);
            return Ok((rem, block));

        // tail
        } else {
            let (rem, tail_expr) = Expression::parse_tail::<E>(rem)?;
            let (rem, _) = tag("}")(rem)?;
            let block = Block {
                tail: Some(tail_expr),
                body,
            };
            debug!("found block: {:#?}", block);
            Ok((rem, block))
        }
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
        trace!("maybe statement: {:?}", i);
        // declarations & blocs
        let maybe = alt((Self::parse_let::<E>, Self::parse_bloc::<E>))(i).ok();
        if let Some((rem, statement)) = maybe {
            debug!("found statement: {:#?}", statement);
            return Ok((rem, statement));
        }

        // naked expressions
        let (mut rem, (expr, _)) = tuple((Expression::parse::<E>, tag(";")))(i)?;
        if rem.len() > 0 {
            let rem_ = opt(basic::sep)(rem)?;
            rem = rem_.0;
        }

        let statement = Statement::Expr { expr };
        debug!("found statement expr: {:#?}", statement);
        return Ok((rem, statement));
    }

    fn parse_bloc<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&str, Statement> {
        let (rem, bloc) = Bloc::parse::<E>(i)?;
        Ok((rem, Statement::Bloc(bloc)))
    }

    fn parse_let<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&str, Statement> {
        let (rem, (_, _, ident)) = tuple((basic::k_let, basic::sep, basic::ident))(i)?;
        let (rem, item_type) = opt(Self::parse_type::<E>)(rem)?;
        let (rem, (_, _, _, expr, _, _)) = tuple((
            basic::sep,
            tag("="),
            basic::sep,
            Expression::parse::<E>,
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
        block: Rc<Block>,
    },
    Loop {
        item: Option<String>,
        iter: Iteration,
        block: Rc<Block>,
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

        Ok((
            rem,
            Bloc::If {
                cond,
                block: Rc::new(block),
            },
        ))
    }

    fn parse_loop<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&str, Bloc> {
        let (rem, maybe_bloc) = opt(Self::parse_loop_infinite::<E>)(i)?;
        if let Some(bloc) = maybe_bloc {
            return Ok((rem, bloc));
        }
        IResult::Err(nom::Err::Error(nom::error::Error::new(
            "todo: finish loop parser",
            nom::error::ErrorKind::Fail,
        )))
    }

    fn parse_loop_infinite<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&str, Bloc> {
        let (rem, (_, _, block)) = tuple((basic::k_loop, basic::sep, Block::parse::<E>))(i)?;
        Ok((
            rem,
            Bloc::Loop {
                item: None,
                iter: Iteration::Infinite,
                block: Rc::new(block),
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
    const TAIL_END_CHARS: &'static str = ";{}";

    fn parse_tail<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&str, Expression> {
        trace!("maybe tail: {:?}", i);
        if i.len() == 0 {
            return Err(nom::Err::Error(nom::error_position!(
                "no expression",
                nom::error::ErrorKind::Fail
            )));
        }

        let (mut rem, outer) = take_while(|c| !Self::TAIL_END_CHARS.contains(c))(i)?;
        let mut expr = Expression {
            outer: outer.to_owned(),
            block: None,
        };

        if rem.len() > 0 && &rem[..1] == "{" {
            let (p_rem, p_block) = Block::parse::<E>(rem)?;
            rem = p_rem;
            expr.block = Some(Rc::new(p_block));
        }

        debug!("found tail: {:#?}", expr);
        Ok((rem, expr))
    }

    pub fn parse<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&str, Expression> {
        trace!("maybe expression: {:?}", i);
        if i.len() == 0 {
            return Err(nom::Err::Error(nom::error_position!(
                "no expression",
                nom::error::ErrorKind::Fail
            )));
        }

        let (mut rem, outer) = take_while(|c| !Self::END_CHARS.contains(c))(i)?;
        let mut expr = Expression {
            outer: outer.to_owned(),
            block: None,
        };

        if rem.len() > 0 && &rem[..1] == "{" {
            let (p_rem, p_block) = Block::parse::<E>(rem)?;
            rem = p_rem;
            expr.block = Some(Rc::new(p_block));
        }

        debug!("found expression: {:#?}", expr);
        Ok((rem, expr))
    }
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

#[derive(Debug)]
pub struct Struct {
    pub ident: String,
    pub public: bool,
    pub entries: Vec<StructEntry>,
    pub attr: Option<Attr>,
}

impl Struct {
    pub fn parse<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&str, Struct> {
        trace!("maybe struct: {:?}", i);
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
