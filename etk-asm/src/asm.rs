//! Single-scope assembler implementation and related types.
//!
//! See [`Assembler`] for more details on the low-level assembly process, or the
//! [`mod@crate::ingest`] module for a higher-level interface.

mod error {
    use crate::ops::Expression;
    use crate::ParseError;
    use etk_ops::cancun::Op;
    use num_bigint::BigInt;
    use snafu::{Backtrace, Snafu};

    /// Errors that can occur while assembling instructions.
    #[derive(Snafu, Debug)]
    #[non_exhaustive]
    #[snafu(context(suffix(false)), visibility(pub(super)))]
    pub enum Error {
        /// A label was declared multiple times.
        #[snafu(display("label `{}` declared multiple times", label))]
        #[non_exhaustive]
        DuplicateLabel {
            /// The name of the conflicting label.
            label: String,

            /// The location of the error.
            backtrace: Backtrace,
        },

        /// A macro was declared multiple times.
        #[snafu(display("macro `{}` declared multiple times", name))]
        #[non_exhaustive]
        DuplicateMacro {
            /// The name of the conflicting macro.
            name: String,

            /// The location of the error.
            backtrace: Backtrace,
        },

        /// A push instruction was too small for the result of the expression.
        #[snafu(display(
            "the expression `{}={}` was too large for the specifier {}",
            expr,
            value,
            spec
        ))]
        #[non_exhaustive]
        ExpressionTooLarge {
            /// The oversized expression.
            expr: Expression,

            /// The evaluated value of the expression.
            value: BigInt,

            /// The specifier.
            spec: Op<()>,

            /// The location of the error.
            backtrace: Backtrace,
        },

        /// An expression evaluated to a negative number.
        #[snafu(display(
            "the expression `{}={}` is negative and can't be represented as push operand",
            expr,
            value
        ))]
        ExpressionNegative {
            /// The oversized expression.
            expr: Expression,

            /// The evaluated value of the expression.
            value: BigInt,

            /// The location of the error.
            backtrace: Backtrace,
        },

        /// The value provided to an unsized push (`%push`) was too large.
        #[snafu(display("value was too large for any push"))]
        #[non_exhaustive]
        UnsizedPushTooLarge {
            /// The location of the error.
            backtrace: Backtrace,
        },

        /// A label was used without being defined.
        #[snafu(display("labels `{:?}` were never defined", labels))]
        #[non_exhaustive]
        UndeclaredLabels {
            /// The labels that were used without being defined.
            labels: Vec<String>,

            /// The location of the error.
            backtrace: Backtrace,
        },

        /// An instruction macro was used without being defined.
        #[snafu(display("instruction macro `{}` was never defined", name))]
        #[non_exhaustive]
        UndeclaredInstructionMacro {
            /// The macro that was used without being defined.
            name: String,

            /// The location of the error.
            backtrace: Backtrace,
        },

        /// An expression macro was used without being defined.
        #[snafu(display("expression macro `{}` was never defined", name))]
        #[non_exhaustive]
        UndeclaredExpressionMacro {
            /// The macro that was used without being defined.
            name: String,

            /// The location of the error.
            backtrace: Backtrace,
        },

        /// An import or include failed to parse.
        #[snafu(display("include or import failed to parse: {}", source))]
        #[snafu(context(false))]
        #[non_exhaustive]
        ParseInclude {
            /// The next source of this error.
            #[snafu(backtrace)]
            source: ParseError,
        },

        /// An instruction macro was used without being defined.
        #[snafu(display("variable `{}` inside macro, was never defined", var))]
        #[non_exhaustive]
        UndeclaredVariableMacro {
            /// The variable that was used without being defined.
            var: String,

            /// The location of the error.
            backtrace: Backtrace,
        },

        /// `--evmgif` was requested but the first emitted instruction is not
        /// `JUMPDEST`. `evmgif` packages the output so the caller jumps to the
        /// byte immediately following the chunk-size marker, which must be a
        /// `JUMPDEST`.
        #[snafu(display(
            "evmgif requires the first instruction to be JUMPDEST (0x5b); found {}",
            actual.map(|b| format!("0x{:02x}", b)).unwrap_or_else(|| "empty output".into())
        ))]
        #[non_exhaustive]
        EvmGifFirstNotJumpdest {
            /// The byte that was actually emitted first, if any.
            actual: Option<u8>,

            /// The location of the error.
            backtrace: Backtrace,
        },

        /// A push referencing a label could not be widened to fit the value
        /// produced by the `--evmgif` transformation (chunk shift + base
        /// offset). The user must request a larger fixed push, or rely on
        /// `%push` so the assembler can choose a size.
        #[snafu(display(
            "evmgif: value 0x{:x} for expression `{}` does not fit into push{}",
            value,
            expr,
            push_size,
        ))]
        #[non_exhaustive]
        EvmGifPushOverflow {
            /// The expression whose evaluated value overflowed the push.
            expr: Expression,

            /// The evaluated value (after applying the chunk shift and base
            /// offset).
            value: BigInt,

            /// The size in bytes of the push immediate that could not hold the
            /// value.
            push_size: usize,

            /// The location of the error.
            backtrace: Backtrace,
        },
    }
}

pub use self::error::Error;
use crate::ops::expression::Error::{UndefinedVariable, UnknownLabel, UnknownMacro};
use crate::ops::{self, AbstractOp, Assemble, Expression, MacroDefinition};
use etk_ops::cancun::Op;
use indexmap::IndexMap;
use num_bigint::{BigInt, Sign};
use rand::Rng;
use std::collections::{hash_map, HashMap, HashSet};

/// EVM opcode for `JUMPDEST`. Used by the `--evmgif` chunker as the entry
/// point inside the first header (`<size> JUMPDEST POP`).
const EVMGIF_JUMPDEST: u8 = 0x5b;

/// EVM opcode for `POP`. Used to consume the size byte that `PUSH1` pushed at
/// the start of each non-initial chunk header.
const EVMGIF_POP: u8 = 0x50;

/// EVM opcode for `PUSH1`. Wraps the size byte in every non-initial chunk
/// header so executing the header is a no-op for the EVM.
const EVMGIF_PUSH1: u8 = 0x60;

/// Maximum number of user-code bytes that may live between two consecutive
/// chunk headers (or between the final header and end-of-output). Combined
/// with the 3-byte header this keeps the inter-header span at most 256 bytes,
/// per the `--evmgif` spec.
const EVMGIF_MAX_CHUNK_CONTENT: usize = 253;

/// Magic bytes that prefix every `--evmgif` payload. The first three bytes
/// (`0x21 0xFF 0x06`) form a GIF Application Extension introducer (extension
/// label `0xFF`, block size `0x06`); the next six spell the ASCII identifier
/// `EVMGIF`. The `--evmgif <offset>` CLI argument is the byte offset at
/// which this `0x21` byte ends up in the wrapping file.
const EVMGIF_MAGIC: [u8; 9] = [0x21, 0xff, 0x06, 0x45, 0x56, 0x4d, 0x47, 0x49, 0x46];

/// An item to be assembled, which can be either an [`AbstractOp`],
/// the inclusion of a new scope or a raw byte sequence.
#[derive(Debug, Clone)]
pub enum RawOp {
    /// An instruction to be assembled.
    Op(AbstractOp),

    /// A new scope to be created with its corresponding list of operations.
    Scope(Vec<RawOp>),

    /// Raw bytes, for example from `%include_hex`, to be included verbatim in
    /// the output.
    Raw(Vec<u8>),
}

impl From<AbstractOp> for RawOp {
    fn from(op: AbstractOp) -> Self {
        Self::Op(op)
    }
}

impl From<Vec<u8>> for RawOp {
    fn from(vec: Vec<u8>) -> Self {
        Self::Raw(vec)
    }
}

impl From<&AbstractOp> for RawOp {
    fn from(op: &AbstractOp) -> Self {
        Self::Op(op.clone())
    }
}

/// Assembles a series of [`RawOp`] into raw bytes, tracking and resolving macros and labels,
/// and handling variable-sized pushes.
///
/// ## Example
///
/// ```rust
/// use etk_asm::asm::Assembler;
/// use etk_asm::ops::AbstractOp;
/// use etk_ops::cancun::{Op, GetPc};
/// # use etk_asm::asm::Error;
/// #
/// # use hex_literal::hex;
/// let mut asm = Assembler::new();
/// let code = vec![AbstractOp::new(GetPc)];
/// let result = asm.assemble(&code)?;
/// # assert_eq!(result, hex!("58"));
/// # Result::<(), Error>::Ok(())
/// ```
#[derive(Debug, Default)]
pub struct Assembler {
    /// Assembled ops.
    ready: Vec<RawOp>,

    /// Number of bytes used by the operations in `ready``.
    concrete_len: usize,

    /// Labels associated with an `AbstractOp::Label`.
    declared_labels: IndexMap<String, Option<LabelDef>>,

    /// Macros associated with an `AbstractOp::Macro`.
    declared_macros: HashMap<String, MacroDefinition>,

    /// Labels that have been referred to (ex. with push) but
    /// have not been declared with an `AbstractOp::Label`.
    undeclared_labels: HashSet<String>,

    /// Pushes that are variable-sized and need to be backpatched.
    variable_sized_push: Vec<PushDef>,

    /// When set, run the `--evmgif` post-pass over the assembled output:
    /// insert chunk-size headers (so the bytecode is also a valid GIF
    /// sub-block stream) and resolve every label-bearing push as
    /// `chunked_position + offset` instead of `position`.
    evmgif_offset: Option<u64>,

    /// Filled during emission only when [`Self::evmgif_offset`] is set:
    /// every push whose immediate references a label, so the evmgif
    /// post-pass can rewrite its bytes once the chunk layout is known.
    evmgif_label_pushes: Vec<EvmGifPushRef>,
}

/// Bookkeeping for a single label-bearing push, captured during emission
/// when `--evmgif` is active. Holds enough state to re-evaluate the push
/// expression against the chunked label positions and write the new bytes
/// in-place.
#[derive(Debug, Clone)]
struct EvmGifPushRef {
    /// Byte offset of the push opcode in the *unchunked* output.
    unchunked_op_pos: usize,

    /// Width of the push immediate (i.e. `push_size - 1`; 1 for `PUSH1`).
    imm_size: usize,

    /// Expression to re-evaluate against the chunked label positions.
    expr: Expression,
}

/// A label definition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LabelDef {
    position: usize,
    updated: bool,
}

impl LabelDef {
    /// Create a new `LabelDef`.
    pub fn new(position: usize) -> Self {
        Self {
            position,
            updated: false,
        }
    }

    /// Get the position of the label.
    pub fn position(&self) -> usize {
        self.position
    }
}

/// A push definition.
#[derive(Clone, Debug, PartialEq)]
pub struct PushDef {
    position: usize,
    op: AbstractOp,
}

impl PushDef {
    /// Create a new `PushDef`.
    pub fn new(op: AbstractOp, position: usize) -> Self {
        Self { op, position }
    }

    /// Get the position of the push.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Get the op from the push.
    pub fn op(&self) -> &AbstractOp {
        &self.op
    }
}

/// Boundaries computed by [`plan_chunks`] — one entry per chunk header that
/// will appear in the chunked output, plus the position where output ends so
/// we can compute the size of the last chunk uniformly.
#[derive(Debug)]
struct ChunkBoundaries {
    /// Byte offsets in the *unchunked* output where each chunk's user-code
    /// content starts. `header_at[0] == 1` (right after the leading
    /// JUMPDEST), and the slice is non-empty because the leading JUMPDEST is
    /// always absorbed into the first header.
    header_at: Vec<usize>,
    /// Total length of the unchunked output. Used as a sentinel "end" so
    /// `header_at` and `unchunked_len` together describe every chunk's
    /// `[start, end)` half-open range.
    unchunked_len: usize,
}

/// Walk `bytes` opcode-by-opcode and return the starting offset of every
/// instruction. Unknown opcodes (e.g. raw bytes from `%include_hex`) are
/// treated as single-byte `InvalidXX` ops, which is the existing
/// disassembler's contract.
fn decode_instruction_starts(bytes: &[u8]) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        starts.push(i);
        let size = Op::<()>::from(bytes[i]).size();
        // A push whose immediate runs past the end of the buffer cannot
        // happen for valid assembler output, but be defensive in case
        // %include_hex produced an unaligned tail.
        i = i.saturating_add(size).min(bytes.len());
    }
    starts
}

/// Decide where to insert chunk headers. The first instruction (always
/// JUMPDEST, checked by the caller) is absorbed into the first header. After
/// that, instructions are packed into chunks of at most
/// [`EVMGIF_MAX_CHUNK_CONTENT`] bytes each, never splitting an instruction.
///
/// Returns one [`ChunkBoundaries::header_at`] entry per chunk; `header_at[0]`
/// is always `1` (the first chunk's user content starts right after the
/// leading JUMPDEST).
fn plan_chunks(instr_starts: &[usize], unchunked_len: usize) -> ChunkBoundaries {
    let mut header_at = vec![1usize];

    if instr_starts.len() <= 1 {
        return ChunkBoundaries {
            header_at,
            unchunked_len,
        };
    }

    let mut current_chunk_start = 1usize;

    // Walk instructions after the leading JUMPDEST (index 0). For each one,
    // either it fits in the current chunk or it starts a new one.
    for idx in 1..instr_starts.len() {
        let instr_start = instr_starts[idx];
        let instr_end = if idx + 1 < instr_starts.len() {
            instr_starts[idx + 1]
        } else {
            unchunked_len
        };
        let instr_size = instr_end - instr_start;

        // If a single instruction is wider than the chunk budget there is
        // nothing we can do — we can't split it. Bail by giving it its own
        // chunk; the resulting size byte will overflow and the caller will
        // see EvmGifPushOverflow further down (or, for non-push wide ops,
        // produce an oversized size byte which we cap below).
        if instr_size > EVMGIF_MAX_CHUNK_CONTENT {
            if current_chunk_start != instr_start {
                header_at.push(instr_start);
            }
            current_chunk_start = instr_start;
            // Force the next instruction (if any) into a fresh chunk too.
            continue;
        }

        let bytes_in_current = instr_end - current_chunk_start;
        if bytes_in_current > EVMGIF_MAX_CHUNK_CONTENT {
            header_at.push(instr_start);
            current_chunk_start = instr_start;
        }
    }

    ChunkBoundaries {
        header_at,
        unchunked_len,
    }
}

/// Build a lookup from unchunked offsets to chunked offsets. Only positions
/// that can appear in a label or push (instruction starts plus the leading
/// JUMPDEST at 0) are entered, since intermediate bytes never appear in any
/// resolved expression.
fn build_position_map(
    instr_starts: &[usize],
    boundaries: &ChunkBoundaries,
) -> HashMap<usize, usize> {
    let mut map = HashMap::new();
    // The 9-byte magic header sits before `<size_0>`. The leading JUMPDEST
    // therefore lands at chunked offset 9 + 1 == 10, right after the
    // initial `<size>` byte, before the inserted POP.
    map.insert(0, EVMGIF_MAGIC.len() + 1);

    // Every other byte sits past two header bytes that don't exist in the
    // unchunked layout (`<size>` and the POP after the JUMPDEST), plus three
    // bytes for each non-initial header that precedes it, plus the constant
    // 9-byte magic prefix that opens the file.
    let mut chunk_idx = 0usize;
    for &start in instr_starts.iter().skip(1) {
        while chunk_idx + 1 < boundaries.header_at.len()
            && start >= boundaries.header_at[chunk_idx + 1]
        {
            chunk_idx += 1;
        }
        let shift = EVMGIF_MAGIC.len() + 2 + 3 * chunk_idx;
        map.insert(start, start + shift);
    }
    map
}

/// Construct the chunked output: leading `<size_0> JUMPDEST POP`, then each
/// chunk's user bytes preceded (for chunks ≥ 1) by `PUSH1 <size_N> POP`.
fn assemble_chunked(unchunked: &[u8], boundaries: &ChunkBoundaries) -> Vec<u8> {
    let headers = &boundaries.header_at;
    let mut out =
        Vec::with_capacity(EVMGIF_MAGIC.len() + unchunked.len() + 3 * headers.len());

    // Every `--evmgif` payload opens with the 9-byte magic identifier so
    // the wrapping file can distinguish embedded EVM payloads from other
    // GIF application extensions.
    out.extend_from_slice(&EVMGIF_MAGIC);

    // Compute each chunk's `[start, end)` range in unchunked space so we can
    // size the headers in one pass.
    let chunk_end = |idx: usize| -> usize {
        if idx + 1 < headers.len() {
            headers[idx + 1]
        } else {
            boundaries.unchunked_len
        }
    };

    for (idx, &chunk_start) in headers.iter().enumerate() {
        let end = chunk_end(idx);
        let content_len = end - chunk_start;

        if idx == 0 {
            // First header: <size> JUMPDEST POP. The JUMPDEST byte is taken
            // from the user's first instruction (already validated to be
            // 0x5b), so we don't copy `unchunked[0]` again into the chunk
            // body — it lives inside the header.
            //
            // `<size>` counts the bytes that follow it before the next
            // header (or end-of-output): JUMPDEST + POP + content_len.
            let size = (2 + content_len) as u8;
            out.push(size);
            out.push(EVMGIF_JUMPDEST);
            out.push(EVMGIF_POP);
        } else {
            // Subsequent headers: PUSH1 <size> POP. `<size>` counts the
            // bytes after itself up to the next header (POP + content_len),
            // which is at most 254 — within byte range.
            let size = (1 + content_len) as u8;
            out.push(EVMGIF_PUSH1);
            out.push(size);
            out.push(EVMGIF_POP);
        }

        out.extend_from_slice(&unchunked[chunk_start..end]);
    }

    out
}

impl Assembler {
    /// Create a new `Assembler`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new `Assembler` in `--evmgif` mode at the given base offset.
    ///
    /// The output is chunked with `<size> JUMPDEST POP` as the first header
    /// and `PUSH1 <size> POP` between subsequent chunks, and every label
    /// resolves to `offset + chunked_position`. The first user instruction
    /// must be `JUMPDEST` — it is absorbed into the first header and used as
    /// the caller's entry point at `offset + 1`.
    pub fn with_evmgif_offset(offset: u64) -> Self {
        Self {
            evmgif_offset: Some(offset),
            ..Default::default()
        }
    }

    /// Feed instructions into the `Assembler`.
    ///
    /// Returns the code of the assembled program.
    pub fn assemble<O>(&mut self, ops: &[O]) -> Result<Vec<u8>, Error>
    where
        O: Into<RawOp> + Clone,
    {
        self.declare_macros(ops)?;

        for op in ops {
            self.push(op.clone().into())?;
        }

        let output = self.backpatch_and_emit()?;
        self.ready.clear();
        Ok(output)
    }

    /// Pre-define macros, via `AbstractOp`, into the `Assembler`.
    ///
    /// This is used to define macros that are used in the same scope.
    fn declare_macros<O>(&mut self, ops: &[O]) -> Result<(), Error>
    where
        O: Into<RawOp> + Clone,
    {
        for op in ops {
            let rop = op.clone().into();
            if let RawOp::Op(AbstractOp::MacroDefinition(ref defn)) = rop {
                match self.declared_macros.entry(defn.name().to_owned()) {
                    hash_map::Entry::Occupied(_) => {
                        return error::DuplicateMacro { name: defn.name() }.fail()
                    }
                    hash_map::Entry::Vacant(v) => {
                        v.insert(defn.to_owned());
                    }
                }
            }
        }

        Ok(())
    }

    /// Feed a single instruction into the `Assembler`.
    fn push<O>(&mut self, rop: O) -> Result<usize, Error>
    where
        O: Into<RawOp>,
    {
        let rop = rop.into();
        self.declare_label(&rop)?;

        match rop {
            RawOp::Op(AbstractOp::Label(label)) => {
                self.undeclared_labels.retain(|l| *l != label);

                let old = self
                    .declared_labels
                    .insert(
                        label,
                        Some(LabelDef {
                            position: self.concrete_len,
                            updated: false,
                        }),
                    )
                    .expect("label should exist");
                assert_eq!(old, None, "label should have been undefined");
            }
            RawOp::Op(AbstractOp::MacroDefinition(_)) => {}
            RawOp::Op(AbstractOp::Macro(ref m)) => {
                self.expand_macro(&m.name, &m.parameters)?;
            }
            RawOp::Op(ref op) => {
                match op
                    .clone()
                    .concretize((&self.declared_labels, &self.declared_macros).into())
                {
                    Ok(cop) => {
                        self.concrete_len += cop.size();
                        self.ready.push(rop.clone())
                    }
                    Err(ops::Error::ExpressionTooLarge { value, spec, .. }) => {
                        return error::ExpressionTooLarge {
                            expr: op.expr().unwrap().clone(),
                            value,
                            spec,
                        }
                        .fail()
                    }
                    Err(ops::Error::ExpressionNegative { value, .. }) => {
                        return error::ExpressionNegative {
                            expr: op.expr().unwrap().clone(),
                            value,
                        }
                        .fail()
                    }
                    Err(ops::Error::ContextIncomplete {
                        source: UnknownLabel { .. },
                    }) => {
                        let labels = op
                            .expr()
                            .unwrap()
                            .labels(&self.declared_macros)
                            .unwrap()
                            .into_iter()
                            .collect::<Vec<String>>();

                        if let AbstractOp::Push(_) = op {
                            // Here, we set the size of the push to 2 bytes (min possible value),
                            //  as we don't know the final value of the label yet.
                            self.concrete_len += 2;
                            self.variable_sized_push.push(PushDef {
                                position: self.concrete_len,
                                op: op.clone(),
                            });
                        } else {
                            self.concrete_len += op.size().unwrap();
                        }

                        self.undeclared_labels.extend(labels);
                        self.ready.push(rop.clone());
                    }
                    Err(ops::Error::ContextIncomplete {
                        source: UnknownMacro { name, .. },
                    }) => return error::UndeclaredInstructionMacro { name }.fail(),
                    Err(ops::Error::ContextIncomplete {
                        source: UndefinedVariable { name, .. },
                    }) => return error::UndeclaredVariableMacro { var: name }.fail(),
                }
            }
            RawOp::Raw(raw) => {
                self.concrete_len += raw.len();
                self.ready.push(RawOp::Raw(raw.to_vec()));
            }
            RawOp::Scope(scope) => {
                let mut asm = Self::new();
                let scope_result = asm.assemble(&scope)?;
                self.concrete_len += scope_result.len();
                self.ready.push(RawOp::Raw(scope_result));
            }
        }

        Ok(self.concrete_len)
    }

    fn backpatch_labels(&mut self) -> Result<(), Error> {
        for pushdef in self.variable_sized_push.iter() {
            if let AbstractOp::Push(imm) = &pushdef.op {
                let exp = imm
                    .tree
                    .eval_with_context((&self.declared_labels, &self.declared_macros).into());

                if let Ok(val) = exp {
                    let val_bits = BigInt::bits(&val).max(1);
                    let imm_size = 1 + ((val_bits - 1) / 8);

                    if imm_size > 1 {
                        for label_value in self.declared_labels.values_mut() {
                            let labeldef = label_value.as_ref().unwrap();
                            if labeldef.position < pushdef.position {
                                // don't move labels that are declared earlier than this push
                                continue;
                            };
                            self.concrete_len += imm_size as usize - 1;

                            *label_value = Some(LabelDef {
                                position: labeldef.position + imm_size as usize - 1,
                                updated: true,
                            });
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Backpatch variable-sized operations and emit the assembled program.
    ///
    /// This function performs the final steps in the assembly process. It ensures that all labels
    /// and variable-sized ops in the code have been properly resolved and finalized. This includes
    /// handling variable-sized push instructions, where the actual size of the push may not be known
    /// until all labels and expressions have been evaluated.
    ///
    /// Handle Variable-sized Pushes: The size of a push operation may depend on the value being pushed, especially
    /// when labels are involved. As labels could be resolved to different addresses during the
    /// assembly process, the final value of a label (and thus the size of the push) might only be
    /// known at this stage. This function recalculates the size of each push operation based on the
    /// final resolved values of labels and expressions. If a push operation requires more space than
    /// initially estimated, the function adjusts the code accordingly.
    fn backpatch_and_emit(&mut self) -> Result<Vec<u8>, Error> {
        if !self.undeclared_labels.is_empty() {
            return error::UndeclaredLabels {
                labels: self
                    .undeclared_labels
                    .iter()
                    .map(|l| l.to_owned())
                    .collect::<Vec<String>>(),
            }
            .fail();
        }
        self.backpatch_labels()?;
        let output = match self.emit_bytecode() {
            Ok(value) => value,
            Err(value) => return value,
        };

        if self.evmgif_offset.is_some() {
            return self.evmgif_chunkify(output);
        }

        Ok(output)
    }

    fn emit_bytecode(&mut self) -> Result<Vec<u8>, Result<Vec<u8>, Error>> {
        let mut output = Vec::new();
        let track_evmgif = self.evmgif_offset.is_some();
        if track_evmgif {
            self.evmgif_label_pushes.clear();
        }
        for op in self.ready.iter() {
            let op = match op {
                RawOp::Op(ref op) => op,
                RawOp::Raw(raw) => {
                    output.extend(raw);
                    continue;
                }
                RawOp::Scope(_) => unreachable!("scopes should be expanded"),
            };

            // In evmgif mode, capture the expression of any push that
            // references at least one label, *before* we concretize and
            // throw away the abstract form. The position is recorded
            // post-emit so we know exactly where the immediate landed.
            let push_with_labels = if track_evmgif {
                op.expr().and_then(|expr| {
                    let labels = expr.labels(&self.declared_macros).ok()?;
                    if labels.is_empty() {
                        None
                    } else {
                        Some(expr.clone())
                    }
                })
            } else {
                None
            };

            let pre_len = output.len();
            match op
                .clone()
                .concretize((&self.declared_labels, &self.declared_macros).into())
            {
                Ok(cop) => cop.assemble(&mut output),
                Err(ops::Error::ContextIncomplete {
                    source: UnknownLabel { .. },
                }) => {
                    return Err(error::UndeclaredLabels {
                        labels: self.undeclared_labels.iter().cloned().collect::<Vec<_>>(),
                    }
                    .fail());
                }
                Err(ops::Error::ContextIncomplete {
                    source: UnknownMacro { name, .. },
                }) => {
                    return Err(error::UndeclaredInstructionMacro { name }.fail());
                }
                Err(ops::Error::ContextIncomplete {
                    source: UndefinedVariable { name, .. },
                }) => {
                    return Err(error::UndeclaredVariableMacro { var: name }.fail());
                }
                Err(_) => unreachable!("all ops should be concretizable"),
            }

            if let Some(expr) = push_with_labels {
                let emitted = output.len() - pre_len;
                debug_assert!(emitted >= 2, "label-bearing push must emit opcode + immediate");
                self.evmgif_label_pushes.push(EvmGifPushRef {
                    unchunked_op_pos: pre_len,
                    imm_size: emitted - 1,
                    expr,
                });
            }
        }
        Ok(output)
    }

    /// Transform the just-emitted unchunked bytecode into the `--evmgif`
    /// layout:
    ///
    /// ```text
    /// 0x21 0xFF 0x06 'E' 'V' 'M' 'G' 'I' 'F'     ; 9-byte magic header
    /// <size_0> JUMPDEST POP <chunk_0 user bytes ...>
    /// PUSH1 <size_1> POP <chunk_1 user bytes ...>
    /// PUSH1 <size_2> POP <chunk_2 user bytes ...>
    /// ...
    /// ```
    ///
    /// The CLI `--evmgif <offset>` argument is the byte position at which
    /// the leading `0x21` magic byte ends up in the wrapping file. The
    /// `JUMPDEST` the caller jumps to therefore lives at `offset + 10`.
    /// Each `<size_N>` byte stores the number of bytes that follow it up to
    /// the next header (or end-of-output). The first header omits `PUSH1`
    /// because the caller jumps directly to the `JUMPDEST`, so the leading
    /// `<size_0>` byte is never executed linearly.
    ///
    /// Labels resolve to `offset + chunked_position`, so every push captured
    /// in [`Self::evmgif_label_pushes`] is rewritten in place. Pushes whose
    /// new value would not fit their original immediate width produce an
    /// [`Error::EvmGifPushOverflow`]; the caller can switch to `%push` or a
    /// larger `pushN` to give the assembler room.
    fn evmgif_chunkify(&mut self, unchunked: Vec<u8>) -> Result<Vec<u8>, Error> {
        let offset = self
            .evmgif_offset
            .expect("evmgif_chunkify called without evmgif_offset");

        // The caller jumps to `offset + 1`, expecting a `JUMPDEST` there. The
        // only byte we can position at offset+1 without disturbing the user's
        // code is the user's first instruction, so it must be a JUMPDEST.
        let first = unchunked.first().copied();
        if first != Some(EVMGIF_JUMPDEST) {
            return error::EvmGifFirstNotJumpdest { actual: first }.fail();
        }

        // 1. Walk instructions to find their starting offsets. The output is
        // EVM bytecode, so `Op::<()>::from(byte).size()` gives the encoded
        // width of each instruction (1 + immediate bytes, including for
        // pushes). Every byte is some opcode (unknowns become `InvalidXX`
        // with size 1), so this never fails mid-stream — it just consumes
        // raw `%include_hex` bytes opcode-by-opcode, which is acceptable.
        let instr_starts = decode_instruction_starts(&unchunked);

        // 2. Partition instructions after the leading JUMPDEST into chunks.
        // The leading JUMPDEST is absorbed into the first header — chunk 0's
        // content therefore starts at instruction index 1.
        let chunk_boundaries = plan_chunks(&instr_starts, unchunked.len());

        // 3. Build the mapping `unchunked offset -> chunked offset` for the
        // pivotal positions we care about: instruction starts and label
        // positions. Non-instruction offsets inside a multi-byte push never
        // appear in a label, so we only need to handle starts.
        let position_map = build_position_map(&instr_starts, &chunk_boundaries);

        // 4. Build the chunked output, copying user bytes verbatim and
        // inserting headers at chunk boundaries.
        let chunked = assemble_chunked(&unchunked, &chunk_boundaries);

        // 5. Update declared labels to their chunked + offset positions so
        // we can re-evaluate the expressions of label-bearing pushes.
        for value in self.declared_labels.values_mut() {
            if let Some(ref mut def) = value {
                let chunked_pos = position_map
                    .get(&def.position)
                    .copied()
                    .expect("label position must be an instruction boundary");
                def.position = chunked_pos + offset as usize;
                def.updated = true;
            }
        }

        // 6. Re-evaluate each tracked label-push and rewrite the immediate
        // in place. The push's chunked location comes from the same mapping
        // we built for labels — every push opcode sits at an instruction
        // boundary by construction.
        let mut chunked = chunked;
        for push in self.evmgif_label_pushes.drain(..) {
            let chunked_op_pos = position_map
                .get(&push.unchunked_op_pos)
                .copied()
                .expect("push opcode must land on an instruction boundary");
            let imm_start = chunked_op_pos + 1;
            let imm_end = imm_start + push.imm_size;

            let value = push
                .expr
                .eval_with_context((&self.declared_labels, &self.declared_macros).into())
                .expect("labels resolved during emit must still resolve");

            let (sign, bytes) = value.to_bytes_be();
            if sign == Sign::Minus {
                // Original emit already rejects negative pushes; re-add
                // `offset` (which is non-negative) cannot introduce one.
                unreachable!("evmgif label push evaluated negative");
            }

            if bytes.len() > push.imm_size {
                return error::EvmGifPushOverflow {
                    expr: push.expr.clone(),
                    value,
                    push_size: push.imm_size,
                }
                .fail();
            }

            // Big-endian, right-justified into the immediate slot. The slot
            // was already zeroed during the initial emit, but we overwrite
            // the leading zeros too in case the value shrank to fewer bytes.
            let pad = push.imm_size - bytes.len();
            for b in chunked[imm_start..imm_start + pad].iter_mut() {
                *b = 0;
            }
            chunked[imm_start + pad..imm_end].copy_from_slice(&bytes);
        }

        Ok(chunked)
    }

    fn declare_label(&mut self, rop: &RawOp) -> Result<(), Error> {
        if let RawOp::Op(AbstractOp::Label(label)) = rop {
            if self.declared_labels.contains_key(label) {
                return error::DuplicateLabel {
                    label: label.to_owned(),
                }
                .fail();
            }
            self.declared_labels.insert(label.to_owned(), None);
        }
        Ok(())
    }

    fn expand_macro(
        &mut self,
        name: &str,
        parameters: &[Expression],
    ) -> Result<Option<usize>, Error> {
        // Remap labels to macro scope.
        match self.declared_macros.get(name).cloned() {
            Some(MacroDefinition::Instruction(mut m)) => {
                if m.parameters.len() != parameters.len() {
                    panic!("invalid number of parameters for macro {}", name);
                }

                let parameters: HashMap<String, Expression> = m
                    .parameters
                    .into_iter()
                    .zip(parameters.iter().cloned())
                    .collect();

                let mut labels = HashMap::<String, String>::new();
                let mut rng = rand::thread_rng();

                // First pass, find locally defined labels and rename them.
                for op in m.contents.iter_mut() {
                    match op {
                        AbstractOp::Label(ref mut label) => {
                            let mangled = format!("{}_{}_{}", m.name, label, rng.gen::<u64>());
                            let old = labels.insert(label.to_owned(), mangled.clone());
                            if old.is_some() {
                                return error::DuplicateLabel {
                                    label: label.to_string(),
                                }
                                .fail();
                            }
                            *label = mangled;
                        }
                        _ => continue,
                    }
                }

                // Second pass, update local label invocations.
                for op in m.contents.iter_mut() {
                    if let Some(expr) = op.expr_mut() {
                        for lbl in expr.labels(&self.declared_macros).unwrap() {
                            if labels.contains_key(&lbl) {
                                expr.replace_label(&lbl, &labels[&lbl]);
                            }
                        }
                    }

                    // Attempt to fill in parameters
                    if let Some(expr) = op.expr_mut() {
                        for (name, value) in parameters.iter() {
                            expr.fill_variable(name, value)
                        }
                    }
                }

                for op in m.contents.iter() {
                    self.push(op)?;
                }
                Ok(Some(self.concrete_len))
            }
            _ => error::UndeclaredInstructionMacro { name }.fail(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{
        Expression, ExpressionMacroDefinition, ExpressionMacroInvocation, Imm,
        InstructionMacroDefinition, InstructionMacroInvocation, Terminal,
    };
    use assert_matches::assert_matches;
    use etk_ops::cancun::*;
    use hex_literal::hex;
    use num_bigint::{BigInt, Sign};

    #[test]
    fn assemble_variable_push_const_while_pending() -> Result<(), Error> {
        let mut asm = Assembler::new();
        let code = vec![
            AbstractOp::Op(Push1(Imm::with_label("label1")).into()),
            AbstractOp::Push(Terminal::Number(0xaabb.into()).into()),
            AbstractOp::Label("label1".into()),
        ];
        let result = asm.assemble(&code)?;
        assert_eq!(result, hex!("600561aabb"));
        Ok(())
    }

    #[test]
    fn assemble_variable_pushes_abab() -> Result<(), Error> {
        let mut asm = Assembler::new();
        let code = vec![
            AbstractOp::new(JumpDest),
            AbstractOp::Push(Imm::with_label("label1")),
            AbstractOp::Push(Imm::with_label("label2")),
            AbstractOp::Label("label1".into()),
            AbstractOp::new(GetPc),
            AbstractOp::Label("label2".into()),
            AbstractOp::new(GetPc),
        ];
        let result = asm.assemble(&code)?;
        assert_eq!(result, hex!("5b600560065858"));
        Ok(())
    }

    #[test]
    fn assemble_variable_pushes_abba() -> Result<(), Error> {
        let mut asm = Assembler::new();
        let code = vec![
            AbstractOp::new(JumpDest),
            AbstractOp::Push(Imm::with_label("label1")),
            AbstractOp::Push(Imm::with_label("label2")),
            AbstractOp::Label("label2".into()),
            AbstractOp::new(GetPc),
            AbstractOp::Label("label1".into()),
            AbstractOp::new(GetPc),
        ];
        let result = asm.assemble(&code)?;
        assert_eq!(result, hex!("5b600660055858"));
        Ok(())
    }

    #[test]
    fn assemble_variable_push1_multiple() -> Result<(), Error> {
        let mut asm = Assembler::new();
        let code = vec![
            AbstractOp::new(JumpDest),
            AbstractOp::Push(Imm::with_label("auto")),
            AbstractOp::Push(Imm::with_label("auto")),
            AbstractOp::Label("auto".into()),
        ];
        let result = asm.assemble(&code)?;
        assert_eq!(result, hex!("5b60056005"));
        Ok(())
    }

    #[test]
    fn assemble_variable_push_const() -> Result<(), Error> {
        let mut asm = Assembler::new();
        let code = vec![AbstractOp::Push(
            Terminal::Number((0x00aaaaaaaaaaaaaaaaaaaaaaaa as u128).into()).into(),
        )];
        let result = asm.assemble(&code)?;
        assert_eq!(result, hex!("6baaaaaaaaaaaaaaaaaaaaaaaa"));
        Ok(())
    }

    #[test]
    fn assemble_variable_push_too_large() {
        let v = BigInt::from_bytes_be(Sign::Plus, &[1u8; 33]);

        let mut asm = Assembler::new();
        let code = vec![AbstractOp::Push(Terminal::Number(v).into())];
        let err = asm.assemble(&code).unwrap_err();

        assert_matches!(err, Error::ExpressionTooLarge { .. });
    }

    #[test]
    fn assemble_variable_push_negative() {
        let mut asm = Assembler::new();
        let code = vec![AbstractOp::Push(Terminal::Number((-1).into()).into())];
        let err = asm.assemble(&code).unwrap_err();

        assert_matches!(err, Error::ExpressionNegative { .. });
    }

    #[test]
    fn assemble_variable_push_const0() -> Result<(), Error> {
        let mut asm = Assembler::new();
        let code = vec![AbstractOp::Push(
            Terminal::Number((0x00 as u128).into()).into(),
        )];
        let result = asm.assemble(&code)?;
        assert_eq!(result, hex!("6000"));
        Ok(())
    }

    #[test]
    fn assemble_variable_push1_known() -> Result<(), Error> {
        let mut asm = Assembler::new();
        let code = vec![
            AbstractOp::new(JumpDest),
            AbstractOp::Label("auto".into()),
            AbstractOp::Push(Imm::with_label("auto")),
        ];
        let result = asm.assemble(&code)?;
        assert_eq!(result, hex!("5b6001"));
        Ok(())
    }

    #[test]
    fn assemble_variable_push1() -> Result<(), Error> {
        let mut asm = Assembler::new();
        let code = vec![
            AbstractOp::Push(Imm::with_label("auto")),
            AbstractOp::Label("auto".into()),
            AbstractOp::new(JumpDest),
        ];
        let result = asm.assemble(&code)?;
        assert_eq!(result, hex!("60025b"));
        Ok(())
    }

    #[test]
    fn assemble_variable_push1_reuse() -> Result<(), Error> {
        let mut asm = Assembler::new();
        let code = vec![
            AbstractOp::Push(Imm::with_label("auto")),
            AbstractOp::Label("auto".into()),
            AbstractOp::new(JumpDest),
            AbstractOp::new(Push1(Imm::with_label("auto"))),
        ];
        let result = asm.assemble(&code)?;
        assert_eq!(result, hex!("60025b6002"));
        Ok(())
    }

    #[test]
    fn assemble_variable_push2() -> Result<(), Error> {
        let mut code = vec![];
        code.push(AbstractOp::Push(Imm::with_label("auto")));
        for _ in 0..255 {
            code.push(AbstractOp::new(GetPc));
        }

        code.push(AbstractOp::Label("auto".into()));
        code.push(AbstractOp::new(JumpDest));

        let mut asm = Assembler::new();
        let result = asm.assemble(&code)?;

        let mut expected = vec![0x61, 0x01, 0x02];
        expected.extend_from_slice(&[0x58; 255]);
        expected.push(0x5b);
        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn assemble_variable_push3() -> Result<(), Error> {
        let mut code = vec![];
        code.push(AbstractOp::Push(Imm::with_label("auto")));
        for _ in 0..65537 {
            code.push(AbstractOp::new(GetPc));
        }

        code.push(AbstractOp::Label("auto".into()));
        code.push(AbstractOp::new(JumpDest));

        let mut asm = Assembler::new();
        let result = asm.assemble(&code)?;

        let mut expected = vec![0x62, 0x01, 0x00, 0x05];
        expected.extend_from_slice(&[0x58; 65537]);
        expected.push(0x5b);

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn assemble_undeclared_label() -> Result<(), Error> {
        let mut asm = Assembler::new();
        let code = vec![AbstractOp::new(Push1(Imm::with_label("hi")))];
        let err = asm.assemble(&code).unwrap_err();
        assert_matches!(err, Error::UndeclaredLabels { labels, .. } if labels == vec!["hi"]);
        Ok(())
    }

    #[test]
    fn assemble_jumpdest_no_label() -> Result<(), Error> {
        let mut asm = Assembler::new();
        let code = vec![AbstractOp::new(JumpDest)];
        let result = asm.assemble(&code)?;
        assert!(asm.declared_labels.is_empty());
        assert_eq!(result, hex!("5b"));
        Ok(())
    }

    #[test]
    fn assemble_jumpdest_with_label() -> Result<(), Error> {
        let mut asm = Assembler::new();
        let ops = vec![AbstractOp::Label("lbl".into()), AbstractOp::new(JumpDest)];

        let result = asm.assemble(&ops)?;
        assert_eq!(asm.declared_labels.len(), 1);
        assert_eq!(
            asm.declared_labels.get("lbl"),
            Some(&Some(LabelDef {
                position: 0,
                updated: false
            }))
        );
        assert_eq!(result, hex!("5b"));
        Ok(())
    }

    #[test]
    fn assemble_jumpdest_jump_with_label() -> Result<(), Error> {
        let ops = vec![
            AbstractOp::Label("lbl".into()),
            AbstractOp::new(JumpDest),
            AbstractOp::new(Push1(Imm::with_label("lbl"))),
        ];

        let mut asm = Assembler::new();
        let result = asm.assemble(&ops)?;
        assert_eq!(result, hex!("5b6000"));

        Ok(())
    }

    #[test]
    fn assemble_labeled_pc() -> Result<(), Error> {
        let ops = vec![
            AbstractOp::new(Push1(Imm::with_label("lbl"))),
            AbstractOp::Label("lbl".into()),
            AbstractOp::new(GetPc),
        ];

        let mut asm = Assembler::new();
        let result = asm.assemble(&ops)?;
        assert_eq!(result, hex!("600258"));

        Ok(())
    }

    #[test]
    fn assemble_jump_jumpdest_with_label() -> Result<(), Error> {
        let ops = vec![
            AbstractOp::new(Push1(Imm::with_label("lbl"))),
            AbstractOp::Label("lbl".into()),
            AbstractOp::new(JumpDest),
        ];

        let mut asm = Assembler::new();
        let result = asm.assemble(&ops)?;
        assert_eq!(result, hex!("60025b"));

        Ok(())
    }

    #[test]
    fn assemble_label_too_large() {
        let mut ops: Vec<_> = vec![AbstractOp::new(GetPc); 255];
        ops.push(AbstractOp::Label("b".into()));
        ops.push(AbstractOp::new(JumpDest));
        ops.push(AbstractOp::Label("a".into()));
        ops.push(AbstractOp::new(JumpDest));
        ops.push(AbstractOp::new(Push1(Imm::with_label("a"))));
        let mut asm = Assembler::new();
        let err = asm.assemble(&ops).unwrap_err();
        assert_matches!(err, Error::ExpressionTooLarge { expr: Expression::Terminal(Terminal::Label(label)), .. } if label == "a");
    }

    #[test]
    fn assemble_label_just_right() -> Result<(), Error> {
        let mut ops: Vec<_> = vec![AbstractOp::new(GetPc); 255];
        ops.push(AbstractOp::Label("b".into()));
        ops.push(AbstractOp::new(JumpDest));
        ops.push(AbstractOp::Label("a".into()));
        ops.push(AbstractOp::new(JumpDest));
        ops.push(AbstractOp::new(Push1(Imm::with_label("b"))));
        let mut asm = Assembler::new();
        let result = asm.assemble(&ops)?;

        let mut expected = vec![0x58; 255];
        expected.push(0x5b);
        expected.push(0x5b);
        expected.push(0x60);
        expected.push(0xff);

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn assemble_instruction_macro_label_underscore() -> Result<(), Error> {
        let ops = vec![
            InstructionMacroDefinition {
                name: "my_macro".into(),
                parameters: vec![],
                contents: vec![AbstractOp::Label("a".into())],
            }
            .into(),
            InstructionMacroDefinition {
                name: "my".into(),
                parameters: vec![],
                contents: vec![AbstractOp::Label("macro_a".into())],
            }
            .into(),
            AbstractOp::Macro(InstructionMacroInvocation {
                name: "my_macro".into(),
                parameters: vec![],
            }),
            AbstractOp::Macro(InstructionMacroInvocation {
                name: "my".into(),
                parameters: vec![],
            }),
        ];

        let mut asm = Assembler::new();
        let result = asm.assemble(&ops)?;
        assert_eq!(result, []);

        Ok(())
    }

    #[test]
    fn assemble_instruction_macro_twice() -> Result<(), Error> {
        let ops = vec![
            InstructionMacroDefinition {
                name: "my_macro".into(),
                parameters: vec![],
                contents: vec![
                    AbstractOp::Label("a".into()),
                    AbstractOp::new(JumpDest),
                    AbstractOp::new(Push1(Imm::with_label("a"))),
                    AbstractOp::new(Push1(Imm::with_label("b"))),
                ],
            }
            .into(),
            AbstractOp::Label("b".into()),
            AbstractOp::new(JumpDest),
            AbstractOp::new(Push1(Imm::with_label("b"))),
            AbstractOp::Macro(InstructionMacroInvocation {
                name: "my_macro".into(),
                parameters: vec![],
            }),
            AbstractOp::Macro(InstructionMacroInvocation {
                name: "my_macro".into(),
                parameters: vec![],
            }),
        ];

        let mut asm = Assembler::new();
        let result = asm.assemble(&ops)?;
        assert_eq!(result, hex!("5b60005b600360005b60086000"));

        Ok(())
    }

    #[test]
    fn assemble_instruction_macro() -> Result<(), Error> {
        let ops = vec![
            InstructionMacroDefinition {
                name: "my_macro".into(),
                parameters: vec![],
                contents: vec![
                    AbstractOp::Label("a".into()),
                    AbstractOp::new(JumpDest),
                    AbstractOp::new(Push1(Imm::with_label("a"))),
                    AbstractOp::new(Push1(Imm::with_label("b"))),
                ],
            }
            .into(),
            AbstractOp::Label("b".into()),
            AbstractOp::new(JumpDest),
            AbstractOp::new(Push1(Imm::with_label("b"))),
            AbstractOp::Macro(InstructionMacroInvocation {
                name: "my_macro".into(),
                parameters: vec![],
            }),
        ];

        let mut asm = Assembler::new();
        let result = asm.assemble(&ops)?;
        assert_eq!(result, hex!("5b60005b60036000"));

        Ok(())
    }

    #[test]
    fn assemble_instruction_macro_delayed_definition() -> Result<(), Error> {
        let ops = vec![
            AbstractOp::Label("b".into()),
            AbstractOp::new(JumpDest),
            AbstractOp::new(Push1(Imm::with_label("b"))),
            AbstractOp::Macro(InstructionMacroInvocation {
                name: "my_macro".into(),
                parameters: vec![],
            }),
            InstructionMacroDefinition {
                name: "my_macro".into(),
                parameters: vec![],
                contents: vec![
                    AbstractOp::Label("a".into()),
                    AbstractOp::new(JumpDest),
                    AbstractOp::new(Push1(Imm::with_label("a"))),
                    AbstractOp::new(Push1(Imm::with_label("b"))),
                ],
            }
            .into(),
        ];

        let mut asm = Assembler::new();
        let result = asm.assemble(&ops)?;
        assert_eq!(result, hex!("5b60005b60036000"));

        Ok(())
    }

    #[test]
    fn assemble_instruction_macro_with_variable_push() -> Result<(), Error> {
        let ops = vec![
            AbstractOp::Macro(InstructionMacroInvocation {
                name: "my_macro".into(),
                parameters: vec![],
            }),
            InstructionMacroDefinition {
                name: "my_macro".into(),
                parameters: vec![],
                contents: vec![
                    AbstractOp::new(JumpDest),
                    AbstractOp::Push(Imm::with_label("label1")),
                    AbstractOp::Push(Imm::with_label("label2")),
                    AbstractOp::Label("label1".into()),
                    AbstractOp::new(GetPc),
                    AbstractOp::Label("label2".into()),
                    AbstractOp::new(GetPc),
                ],
            }
            .into(),
        ];

        let mut asm = Assembler::new();
        let result = asm.assemble(&ops)?;
        assert_eq!(result, hex!("5b600560065858"));

        Ok(())
    }

    #[test]
    fn assemble_undeclared_instruction_macro() -> Result<(), Error> {
        let ops = vec![AbstractOp::Macro(
            InstructionMacroInvocation::with_zero_parameters("my_macro".into()),
        )];
        let mut asm = Assembler::new();
        let err = asm.assemble(&ops).unwrap_err();
        assert_matches!(err, Error::UndeclaredInstructionMacro { name, .. } if name == "my_macro");

        Ok(())
    }

    #[test]
    fn assemble_duplicate_instruction_macro() -> Result<(), Error> {
        let ops: Vec<AbstractOp> = vec![
            InstructionMacroDefinition {
                name: "my_macro".into(),
                parameters: vec![],
                contents: vec![AbstractOp::new(Caller)],
            }
            .into(),
            InstructionMacroDefinition {
                name: "my_macro".into(),
                parameters: vec![],
                contents: vec![AbstractOp::new(Caller)],
            }
            .into(),
        ];
        let mut asm = Assembler::new();
        let err = asm.assemble(&ops).unwrap_err();
        assert_matches!(err, Error::DuplicateMacro { name, .. } if name == "my_macro");

        Ok(())
    }

    #[test]
    fn assemble_duplicate_labels_in_instruction_macro() -> Result<(), Error> {
        let ops = vec![
            InstructionMacroDefinition {
                name: "my_macro".into(),
                parameters: vec![],
                contents: vec![AbstractOp::Label("a".into()), AbstractOp::Label("a".into())],
            }
            .into(),
            AbstractOp::Macro(InstructionMacroInvocation::with_zero_parameters(
                "my_macro".into(),
            )),
        ];
        let mut asm = Assembler::new();
        let err = asm.assemble(&ops).unwrap_err();
        assert_matches!(err, Error::DuplicateLabel { label, .. } if label == "a");

        Ok(())
    }

    // TODO: do we allow label shadowing in macros?
    #[test]
    fn assemble_conflicting_labels_in_instruction_macro() -> Result<(), Error> {
        let ops = vec![
            AbstractOp::Label("a".into()),
            AbstractOp::new(Caller),
            InstructionMacroDefinition {
                name: "my_macro()".into(),
                parameters: vec![],
                contents: vec![
                    AbstractOp::Label("a".into()),
                    AbstractOp::new(Push1(Imm::with_label("a"))),
                ],
            }
            .into(),
            AbstractOp::Macro(InstructionMacroInvocation::with_zero_parameters(
                "my_macro()".into(),
            )),
            AbstractOp::new(Push1(Imm::with_label("a"))),
        ];
        let mut asm = Assembler::new();
        let result = asm.assemble(&ops)?;

        assert_eq!(result, hex!("3360016000"));

        Ok(())
    }

    #[test]
    fn assemble_instruction_macro_with_parameters() -> Result<(), Error> {
        let ops = vec![
            InstructionMacroDefinition {
                name: "my_macro".into(),
                parameters: vec!["foo".into(), "bar".into()],
                contents: vec![
                    AbstractOp::new(Push1(Imm::with_variable("foo"))),
                    AbstractOp::new(Push1(Imm::with_variable("bar"))),
                ],
            }
            .into(),
            AbstractOp::Label("b".into()),
            AbstractOp::new(JumpDest),
            AbstractOp::new(Push1(Imm::with_label("b"))),
            AbstractOp::Macro(InstructionMacroInvocation {
                name: "my_macro".into(),
                parameters: vec![
                    BigInt::from_bytes_be(Sign::Plus, &vec![0x42]).into(),
                    Terminal::Label("b".to_string()).into(),
                ],
            }),
        ];

        let mut asm = Assembler::new();
        let result = asm.assemble(&ops)?;
        assert_eq!(result, hex!("5b600060426000"));

        Ok(())
    }

    #[test]
    fn assemble_expression_push() -> Result<(), Error> {
        let ops = vec![AbstractOp::new(Push1(Imm::with_expression(
            Expression::Plus(1.into(), 1.into()),
        )))];

        let mut asm = Assembler::new();
        let result = asm.assemble(&ops)?;
        assert_eq!(result, hex!("6002"));

        Ok(())
    }

    #[test]
    fn assemble_expression_negative() -> Result<(), Error> {
        let ops = vec![AbstractOp::new(Push1(Imm::with_expression(
            BigInt::from(-1).into(),
        )))];
        let mut asm = Assembler::new();
        let err = asm.assemble(&ops).unwrap_err();
        assert_matches!(err, Error::ExpressionNegative { value, .. } if value == BigInt::from(-1));

        Ok(())
    }

    #[test]
    fn assemble_expression_undeclared_label() -> Result<(), Error> {
        let mut asm = Assembler::new();
        let ops = vec![AbstractOp::new(Push1(Imm::with_expression(
            Terminal::Label(String::from("hi")).into(),
        )))];
        let err = asm.assemble(&ops).unwrap_err();
        assert_matches!(err, Error::UndeclaredLabels { labels, .. } if labels == vec!["hi"]);
        Ok(())
    }

    #[test]
    fn assemble_variable_push_before_push2() -> Result<(), Error> {
        let mut asm = Assembler::new();
        let ops = vec![
            AbstractOp::Push(Imm::with_expression(Expression::Plus(
                Terminal::Label("foo".into()).into(),
                BigInt::from(256).into(),
            ))),
            AbstractOp::new(Push2(Imm::with_label("foo1"))),
            AbstractOp::Label("foo".into()),
            AbstractOp::Label("foo1".into()),
        ];
        let result = asm.assemble(&ops)?;
        assert_eq!(result, hex!("610106610006"));
        Ok(())
    }

    #[test]
    fn assemble_variable_push_before_push2_inverted() -> Result<(), Error> {
        let mut asm = Assembler::new();
        let ops = vec![
            AbstractOp::Push(Imm::with_expression(Expression::Plus(
                Terminal::Label("z".into()).into(),
                BigInt::from(256).into(),
            ))),
            AbstractOp::new(Push2(Imm::with_label("a"))),
            AbstractOp::Label("z".into()),
            AbstractOp::Label("a".into()),
        ];
        let result = asm.assemble(&ops)?;
        assert_eq!(result, hex!("610106610006"));
        Ok(())
    }

    #[test]
    fn assemble_variable_push_mixed_labels() -> Result<(), Error> {
        let mut asm = Assembler::new();
        let ops = vec![
            AbstractOp::Push(Imm::with_expression(Expression::Plus(
                Terminal::Label("foo".into()).into(),
                BigInt::from(256).into(),
            ))),
            AbstractOp::new(Push2(Imm::with_label("foo"))),
            AbstractOp::Push(Imm::with_expression(Expression::Plus(
                Terminal::Label("bar".into()).into(),
                BigInt::from(256).into(),
            ))),
            AbstractOp::new(Gas),
            AbstractOp::Label("foo".into()),
            AbstractOp::new(Gas),
            AbstractOp::Label("bar".into()),
        ];
        let result = asm.assemble(&ops)?;

        assert_eq!(result, hex!("61010a61000a61010b5a5a"));
        Ok(())
    }

    #[test]
    fn assemble_double_update_variable_push() -> Result<(), Error> {
        let mut asm = Assembler::new();
        let ops = vec![
            AbstractOp::Push(Imm::with_expression(Expression::Plus(
                Terminal::Label("foo".into()).into(),
                BigInt::from(256).into(),
            ))),
            AbstractOp::new(Push2(Imm::with_label("foo1"))),
            AbstractOp::Push(Imm::with_expression(Expression::Plus(
                Terminal::Label("foo".into()).into(),
                BigInt::from(256).into(),
            ))),
            AbstractOp::Label("foo".into()),
            AbstractOp::Label("foo1".into()),
        ];
        let result = asm.assemble(&ops)?;
        assert_eq!(result, hex!("610109610009610109"));
        Ok(())
    }

    #[test]
    fn assemble_double_update_variable_push2() -> Result<(), Error> {
        let mut asm = Assembler::new();
        let ops = vec![
            AbstractOp::Push(Imm::with_expression(Expression::Plus(
                Terminal::Label("foo".into()).into(),
                BigInt::from(256).into(),
            ))),
            AbstractOp::new(Push2(Imm::with_label("foo1"))),
            AbstractOp::Push(Imm::with_expression(Expression::Plus(
                Terminal::Label("foo".into()).into(),
                BigInt::from(256).into(),
            ))),
            AbstractOp::new(Push2(Imm::with_label("foo1"))),
            AbstractOp::Label("foo".into()),
            AbstractOp::Label("foo1".into()),
        ];
        let result = asm.assemble(&ops)?;
        assert_eq!(result, hex!("61010c61000c61010c61000c"));
        Ok(())
    }

    #[test]
    fn assemble_variable_push_expression_with_undeclared_labels() -> Result<(), Error> {
        let mut asm = Assembler::new();
        let ops = vec![
            AbstractOp::new(JumpDest),
            AbstractOp::Push(Imm::with_expression(Expression::Plus(
                Terminal::Label("foo".into()).into(),
                Terminal::Label("bar".into()).into(),
            ))),
            AbstractOp::new(Gas),
        ];
        let err = asm.assemble(&ops).unwrap_err();
        // The expressions have short-circuit evaluation, so only the first label is caught in the error.
        assert_matches!(err, Error::UndeclaredLabels { labels, .. } if (labels.contains(&"foo".to_string())));
        Ok(())
    }

    #[test]
    fn assemble_variable_push2_comparison_with_undeclared_labels() -> Result<(), Error> {
        let mut asm = Assembler::new();

        // %push(lbl1 - lbl2)
        // push2 lbl1 + lbl2
        // pc # repeat 126 times.
        // lbl1:
        // lbl2:
        let mut ops = vec![AbstractOp::new(GetPc); 130];
        ops[0] = AbstractOp::Push(Imm::with_expression(Expression::Minus(
            Terminal::Label(String::from("lbl1")).into(),
            Terminal::Label(String::from("lbl2")).into(),
        )));
        ops[1] = AbstractOp::new(Push2(
            Expression::Plus(
                Terminal::Label(String::from("lbl1")).into(),
                Terminal::Label(String::from("lbl2")).into(),
            )
            .into(),
        ));
        ops[128] = AbstractOp::Label("lbl1".into());
        ops[129] = AbstractOp::Label("lbl2".into());

        let expected = asm.assemble(&ops)?;

        let mut asm = Assembler::new();

        // %push(lbl1 - lbl2)
        // %push(lbl1 + lbl2)
        // pc # repeat 126 times.
        // lbl1:
        // lbl2:
        ops[1] = AbstractOp::Push(Imm::with_expression(Expression::Plus(
            Terminal::Label(String::from("lbl1")).into(),
            Terminal::Label(String::from("lbl2")).into(),
        )));
        let result = asm.assemble(&ops)?;

        // Sanity check the expected result: should use push1 then push2.
        assert_eq!(expected[0], 0x60);
        assert_eq!(expected[2], 0x61);

        // Assert that the two results are identical.
        assert_eq!(expected, result);
        Ok(())
    }

    #[test]
    fn assemble_variable_push1_expression() -> Result<(), Error> {
        let mut asm = Assembler::new();
        let ops = vec![
            AbstractOp::new(JumpDest),
            AbstractOp::Label("auto".into()),
            AbstractOp::Push(Imm::with_expression(Expression::Plus(
                1.into(),
                Terminal::Label(String::from("auto")).into(),
            ))),
        ];
        let result = asm.assemble(&ops)?;
        assert_eq!(result, hex!("5b6002"));
        Ok(())
    }

    #[test]
    fn assemble_expression_with_labels() -> Result<(), Error> {
        let mut asm = Assembler::new();
        let ops = vec![
            AbstractOp::new(JumpDest),
            AbstractOp::Push(Imm::with_expression(Expression::Plus(
                Terminal::Label(String::from("foo")).into(),
                Terminal::Label(String::from("bar")).into(),
            ))),
            AbstractOp::new(Gas),
            AbstractOp::Label("foo".into()),
            AbstractOp::Label("bar".into()),
        ];
        let result = asm.assemble(&ops)?;
        assert_eq!(result, hex!("5b60085a"));
        Ok(())
    }

    #[test]
    fn assemble_expression_macro_push() -> Result<(), Error> {
        let ops = vec![
            ExpressionMacroDefinition {
                name: "foo".into(),
                parameters: vec![],
                content: Imm::with_expression(Expression::Plus(1.into(), 1.into())),
            }
            .into(),
            AbstractOp::new(Push1(Imm::with_macro(ExpressionMacroInvocation {
                name: "foo".into(),
                parameters: vec![],
            }))),
        ];

        let mut asm = Assembler::new();
        let result = asm.assemble(&ops)?;
        assert_eq!(result, hex!("6002"));

        Ok(())
    }

    #[test]
    fn assemble_instruction_macro_with_undeclared_variables() {
        let ops = vec![
            InstructionMacroDefinition {
                name: "my_macro".into(),
                parameters: vec!["foo".into()],
                contents: vec![AbstractOp::new(Push1(Imm::with_variable("bar")))],
            }
            .into(),
            AbstractOp::Label("b".into()),
            AbstractOp::new(JumpDest),
            AbstractOp::new(Push1(Imm::with_label("b"))),
            AbstractOp::Macro(InstructionMacroInvocation {
                name: "my_macro".into(),
                parameters: vec![BigInt::from_bytes_be(Sign::Plus, &vec![0x42]).into()],
            }),
        ];

        let mut asm = Assembler::new();
        let err = asm.assemble(&ops).unwrap_err();

        assert_matches!(err, Error::UndeclaredVariableMacro { var, .. } if var == "bar");
    }

    #[test]
    fn assemble_instruction_macro_two_delayed_definitions_mirrored() -> Result<(), Error> {
        let ops = vec![
            AbstractOp::new(GetPc),
            AbstractOp::Macro(InstructionMacroInvocation {
                name: "macro1".into(),
                parameters: vec![],
            }),
            AbstractOp::Macro(InstructionMacroInvocation {
                name: "macro0".into(),
                parameters: vec![],
            }),
            InstructionMacroDefinition {
                name: "macro0".into(),
                parameters: vec![],
                contents: vec![AbstractOp::new(JumpDest)],
            }
            .into(),
            InstructionMacroDefinition {
                name: "macro1".into(),
                parameters: vec![],
                contents: vec![AbstractOp::new(Caller)],
            }
            .into(),
        ];

        let mut asm = Assembler::new();
        let result = asm.assemble(&ops)?;
        assert_eq!(result, hex!("58335b"));

        Ok(())
    }

    // ---------- --evmgif tests ----------

    #[test]
    fn evmgif_rejects_non_jumpdest_start() {
        let mut asm = Assembler::with_evmgif_offset(0);
        // First op is GetPc, not JumpDest -> should fail.
        let ops = vec![AbstractOp::new(GetPc), AbstractOp::new(JumpDest)];
        let err = asm.assemble(&ops).unwrap_err();
        assert_matches!(
            err,
            Error::EvmGifFirstNotJumpdest { actual: Some(0x58), .. }
        );
    }

    #[test]
    fn evmgif_rejects_empty_program() {
        let mut asm = Assembler::with_evmgif_offset(0);
        let ops: Vec<AbstractOp> = vec![];
        let err = asm.assemble(&ops).unwrap_err();
        assert_matches!(err, Error::EvmGifFirstNotJumpdest { actual: None, .. });
    }

    #[test]
    fn evmgif_smallest_program_just_jumpdest() -> Result<(), Error> {
        // The leading JUMPDEST is absorbed into the first header. With no
        // other instructions chunk 0 has zero content, so:
        //   size_0 = 2 (JUMPDEST + POP that follow it)
        let mut asm = Assembler::with_evmgif_offset(0);
        let ops = vec![AbstractOp::new(JumpDest)];
        let result = asm.assemble(&ops)?;
        // 9 magic bytes + <size_0=2> JUMPDEST POP
        assert_eq!(result, hex!("21ff064556 4d474946 025b50"));
        Ok(())
    }

    #[test]
    fn evmgif_short_program_offset_zero() -> Result<(), Error> {
        // Layout: MAGIC | <size_0=4> JUMPDEST POP GAS STOP
        // size_0 counts JUMPDEST + POP + GAS + STOP = 4 bytes.
        let mut asm = Assembler::with_evmgif_offset(0);
        let ops = vec![
            AbstractOp::new(JumpDest),
            AbstractOp::new(Gas),
            AbstractOp::new(Stop),
        ];
        let result = asm.assemble(&ops)?;
        assert_eq!(result, hex!("21ff064556 4d474946 045b505a00"));
        Ok(())
    }

    #[test]
    fn evmgif_label_push_picks_up_offset() -> Result<(), Error> {
        // Layout (unchunked positions): the leading JUMPDEST is at 0, the
        // label `dest` is declared next and points at the *second*
        // JUMPDEST at unchunked pos 1. In the chunked output that lands at
        // EVMGIF_MAGIC.len() (9) + size byte (1) + POP (1) = 12, plus the
        // caller offset 0x10, giving 0x1c.
        let mut asm = Assembler::with_evmgif_offset(0x10);
        let ops = vec![
            AbstractOp::new(JumpDest),
            AbstractOp::Label("dest".into()),
            AbstractOp::new(JumpDest),
            AbstractOp::new(Push1(Imm::with_label("dest"))),
            AbstractOp::new(Jump),
        ];
        let result = asm.assemble(&ops)?;
        // MAGIC | <size_0=6> JUMPDEST POP | JUMPDEST(dest) PUSH1 0x1c JUMP
        let mut expected = EVMGIF_MAGIC.to_vec();
        expected.extend_from_slice(&hex!("065b50"));
        expected.extend_from_slice(&hex!("5b601c56"));
        assert_eq!(result, expected);
        Ok(())
    }

    #[test]
    fn evmgif_inserts_chunk_header_when_content_exceeds_limit() -> Result<(), Error> {
        // 1 JUMPDEST + 254 GAS ops in the unchunked stream. Chunk 0 can hold
        // at most 253 bytes of user content; instruction #254 spills into
        // chunk 1.
        let mut ops: Vec<AbstractOp> = Vec::with_capacity(255);
        ops.push(AbstractOp::new(JumpDest));
        for _ in 0..254 {
            ops.push(AbstractOp::new(Gas));
        }
        let mut asm = Assembler::with_evmgif_offset(0);
        let result = asm.assemble(&ops)?;

        // MAGIC (9) | <size_0 = 2 + 253 = 255> JUMPDEST POP (3)
        // Chunk 0 content: 253 GAS bytes.
        // PUSH1 <size_1 = 1 + 1 = 2> POP (3)
        // Chunk 1 content: 1 GAS byte.
        let mut expected = EVMGIF_MAGIC.to_vec();
        expected.extend_from_slice(&[255u8, EVMGIF_JUMPDEST, EVMGIF_POP]);
        expected.extend(std::iter::repeat(0x5a).take(253));
        expected.extend_from_slice(&[EVMGIF_PUSH1, 0x02, EVMGIF_POP]);
        expected.push(0x5a);
        assert_eq!(result, expected);
        Ok(())
    }

    #[test]
    fn evmgif_label_in_second_chunk_picks_up_chunk_shift() -> Result<(), Error> {
        // The label lands inside the second chunk, so the pushed value
        // includes the 9-byte magic, the +2 prefix shift and the +3
        // second-chunk header. With offset 0, the chunked position of
        // `dest` is: unchunked_pos(dest) + 9 + 2 + 3 = (1 + 253) + 14 = 268.
        let mut ops: Vec<AbstractOp> = Vec::new();
        ops.push(AbstractOp::new(JumpDest));
        for _ in 0..253 {
            ops.push(AbstractOp::new(Gas));
        }
        ops.push(AbstractOp::Label("dest".into()));
        ops.push(AbstractOp::new(JumpDest));
        ops.push(AbstractOp::new(Push2(Imm::with_label("dest"))));

        let mut asm = Assembler::with_evmgif_offset(0);
        let result = asm.assemble(&ops)?;

        // PUSH2 sits at chunked offset (unchunked 255 + 9 + 2 + 3) = 269;
        // the immediate occupies bytes 270..272.
        // 268 == 0x010c.
        assert_eq!(result[270], 0x01);
        assert_eq!(result[271], 0x0c);

        // Sanity-check chunk 1 header sits in the right place: chunk 0
        // spans bytes 12..265 of output; header 1 at 265..268.
        assert_eq!(result[265], EVMGIF_PUSH1);
        // size_1 = 1 (POP) + (final chunk content length)
        // chunk 1 content = JUMPDEST(dest) + PUSH2 + 2 bytes = 4
        assert_eq!(result[266], 0x05);
        assert_eq!(result[267], EVMGIF_POP);

        // And the leading 9 bytes are the magic identifier.
        assert_eq!(&result[..9], &EVMGIF_MAGIC);
        Ok(())
    }

    #[test]
    fn evmgif_push_overflow_when_value_outgrows_immediate() {
        // PUSH1 referencing a label whose chunked + offset position exceeds
        // 255 must fail with EvmGifPushOverflow rather than silently
        // truncating.
        let mut asm = Assembler::with_evmgif_offset(0xff00);
        let ops = vec![
            AbstractOp::new(JumpDest),
            AbstractOp::new(Push1(Imm::with_label("self_ref"))),
            AbstractOp::Label("self_ref".into()),
            AbstractOp::new(JumpDest),
        ];
        let err = asm.assemble(&ops).unwrap_err();
        assert_matches!(err, Error::EvmGifPushOverflow { push_size: 1, .. });
    }

    #[test]
    fn evmgif_prepends_magic_header() -> Result<(), Error> {
        // Regardless of program content, every --evmgif payload starts with
        // the 9-byte magic header.
        let mut asm = Assembler::with_evmgif_offset(0);
        let ops = vec![AbstractOp::new(JumpDest), AbstractOp::new(Stop)];
        let result = asm.assemble(&ops)?;
        assert_eq!(&result[..9], &EVMGIF_MAGIC);
        Ok(())
    }

    #[test]
    fn assemble_instruction_macro_two_delayed_definitions() -> Result<(), Error> {
        let ops = vec![
            AbstractOp::new(GetPc),
            AbstractOp::Macro(InstructionMacroInvocation {
                name: "macro0".into(),
                parameters: vec![],
            }),
            AbstractOp::Macro(InstructionMacroInvocation {
                name: "macro1".into(),
                parameters: vec![],
            }),
            InstructionMacroDefinition {
                name: "macro0".into(),
                parameters: vec![],
                contents: vec![AbstractOp::new(JumpDest)],
            }
            .into(),
            InstructionMacroDefinition {
                name: "macro1".into(),
                parameters: vec![],
                contents: vec![AbstractOp::new(Caller)],
            }
            .into(),
        ];

        let mut asm = Assembler::new();
        let result = asm.assemble(&ops)?;
        assert_eq!(result, hex!("585b33"));

        Ok(())
    }
}
