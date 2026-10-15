//! How a chat template writes a table's turn where the response template's patterns leave the
//! text open.
//!
//! A response template reads more than one spelling of a call: `\s*` takes any whitespace,
//! `[^>]*?` any attributes, and the text outside every field is dropped whatever it is. A
//! structural tag writes one spelling, and it has to be the one the model's chat template renders,
//! or the constraint pushes the model off the text it was trained on. A [`Layout`] is that
//! spelling, as the chat template rendered a turn:
//!
//! - how one call is written ([`CallLayout`]): the text before the call's terminal, the header,
//!   a parameter with its tag, value and end, the text around the parameters, the trailer, and the
//!   terminal that closes the call, each a [`Written`] text with the places where the tool's name,
//!   the key and the value stand;
//! - what comes between two calls and after the last;
//! - how a value that is not a string is spelled ([`Spellings`]);
//! - what a thought and an answer are written with around the table's terminals, and the terminal
//!   that closes each, of the ones the table has ([`Message`]).
//!
//! Whoever compiles a table learns its layout from a rendering of the chat template (a probe
//! conversation, read back through the table) and gives it to the table
//! ([`Format::with_layout`]). Parsing does not read it. [`Layout::check`] says whether it fits the
//! table: the header, a parameter and the trailer as written match the shape's patterns, capturing
//! the name, the key and the value where the layout puts them, and the terminals it names close
//! what it says they close. The structural tags of a shaped call are written from the layout
//! ([`Format::grammar_for`]), and a table's thought and answer as it writes them; a shaped call
//! whose table has no layout that fits has no tag.
//!
//! [`Format::with_layout`]: crate::Format::with_layout
//! [`Format::grammar_for`]: crate::Format::grammar_for

use super::{
    Arguments, Atom, CallShape, Class, Match, Matcher, NameSource, Pattern, Slot, Verdict,
};
use crate::format::{CallSyntax, Emits, Format};

/// A text as the template writes it, with the places where a call's name, a parameter's key and
/// its value stand.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Written(pub Box<[Segment]>);

/// One segment of a [`Written`] text.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Segment {
    /// These characters.
    Lit(Box<str>),
    /// The place of the tool's name, a parameter's key or its value.
    Slot(Slot),
}

impl Written {
    /// A text with no place in it.
    pub fn literal(text: &str) -> Self {
        if text.is_empty() {
            return Self::default();
        }
        Self([Segment::Lit(text.into())].into())
    }

    /// How many places for `slot` the text has.
    fn places(&self, slot: Slot) -> usize {
        self.0
            .iter()
            .filter(|segment| **segment == Segment::Slot(slot))
            .count()
    }

    /// The text with `name` and `key` written in their places, cut where the value stands: what
    /// comes before the value, and what comes after it (empty when the text has no value).
    pub(crate) fn around_the_value(&self, name: &str, key: &str) -> (String, String) {
        let (mut before, mut after) = (String::new(), String::new());
        let mut out = &mut before;
        for segment in &*self.0 {
            match segment {
                Segment::Lit(text) => out.push_str(text),
                Segment::Slot(Slot::Name) => out.push_str(name),
                Segment::Slot(Slot::Key) => out.push_str(key),
                Segment::Slot(Slot::Value) => out = &mut after,
            }
        }
        (before, after)
    }

    /// The text with `name`, `key` and `value` written in their places, and where each stands in
    /// it, by its slot.
    fn filled(&self, name: &str, key: &str, value: &str) -> (String, [Option<(usize, usize)>; 3]) {
        let mut text = String::new();
        let mut spans = [None; 3];
        for segment in &*self.0 {
            let (slot, written) = match segment {
                Segment::Lit(lit) => {
                    text.push_str(lit);
                    continue;
                }
                Segment::Slot(slot) => match slot {
                    Slot::Name => (0, name),
                    Slot::Key => (1, key),
                    Slot::Value => (2, value),
                },
            };
            spans[slot] = Some((text.len(), text.len() + written.len()));
            text.push_str(written);
        }
        (text, spans)
    }
}

/// How a template writes one call of a shaped table, part by part, as its rendering showed: a
/// call is `lead`, the region's terminal, `header`, `before`, the parameters (each as `parameter`
/// says, `between` two of them), `after`, `trailer` and `close`. The text `between` and `after`
/// begin with alike follows every parameter (the newline a template writes after each), so a call
/// that writes no parameter has `before` and then the rest of `after`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct CallLayout {
    /// What the template writes before the call's terminal, where a call is the first thing the
    /// turn writes after the prompt or after a thought: a message's header the response template
    /// reads as text outside every field (Muse Glimmer's ` to=NAME<|message|><atem:function_calls>`
    /// and a newline), or nothing. It may name the tool.
    pub lead: Written,
    /// The header, after the terminal, with the tool's name in its place where the header captures
    /// it (`\n<function=NAME>` for the `\s*<function=(?P<name>…)>` of Qwen 3.5's).
    pub header: Written,
    /// What comes after the header, before the first parameter.
    pub before: Box<str>,
    /// One parameter: its tag with the key and the value in their places, and the end the template
    /// writes after the value (`<parameter=KEY>\nVALUE\n</parameter>`).
    pub parameter: Written,
    /// What comes between two parameters.
    pub between: Box<str>,
    /// What comes after the last parameter, before the trailer.
    pub after: Box<str>,
    /// The trailer, before the closing terminal (`</function>\n`); empty for a shape with none.
    pub trailer: Box<str>,
    /// The terminal that closes each call, or `None` when the stream's stop closes the last one (a
    /// stop token, which the stop decoder hides: the table's stop row), the next call then
    /// following [`Layout::between_calls`], which holds a terminal that closes the call.
    pub close: Option<Box<str>>,
}

/// How a template writes a thought or an answer around the terminals of the rows that open and
/// close it, as its rendering showed.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Message {
    /// What it writes before the terminal that opens the region (Muse Glimmer's space before
    /// `to=user<|message|>`).
    pub before: Box<str>,
    /// What it writes after the region's close, before what comes next (Muse Glimmer's
    /// `<|start|>assistant` after a thought's `<|eom|>`); after an answer the stop ends, what it
    /// writes before the stop.
    pub after: Box<str>,
    /// Whether the stream's stop ends the region (a stop token, which the stop decoder hides)
    /// rather than a terminal the template writes.
    pub stops: bool,
    /// The terminal that closes the region as the rendering wrote it, which the tags write: a
    /// template may close a region more than one way (a response template's `close` given as a
    /// list), and the table's rows do not say which one its chat template renders. `None` where
    /// the stop ends the region, or where the layout does not say: the tags then write the
    /// region's first row out in the table.
    pub close: Option<Box<str>>,
}

/// How a template writes `true`, `false` and `null`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Words {
    /// As JSON writes them.
    #[default]
    Json,
    /// As Python does: `True`, `False`, `None`.
    Python,
}

/// How a template writes a list or an object.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Nested {
    /// As JSON, which the value's schema constrains.
    #[default]
    Json,
    /// As text no schema spells (Python's repr).
    Text,
}

/// How a template writes a value its schema types as other than a string, as its rendering of a
/// call showed. A number is JSON.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Spellings {
    pub words: Words,
    pub nested: Nested,
}

/// How a chat template writes a turn of a table where the response template's patterns leave the
/// text open, as it rendered one: how it writes a call ([`CallLayout`]), what comes between two
/// calls and after the last, how it spells a value that is not a string ([`Spellings`]), and what
/// it writes around a thought and an answer ([`Message`]).
///
/// A response template reads more than one spelling (`\s*` takes any whitespace, the text outside
/// every field is dropped whatever it is); a structural tag writes one, and it has to be the one
/// the chat template renders, or the constraint pushes the model off the text it was trained on.
/// Whoever compiles a table learns its layout from a rendering of the chat template and gives it
/// to the table ([`Format::with_layout`](crate::Format::with_layout)); [`Layout::check`] says
/// whether it fits. Parsing does not read it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Layout {
    /// How it writes one call.
    pub call: CallLayout,
    /// What it writes after one call (after its close, when it writes one) and before the next
    /// call's lead, or `None` when it writes one call in a turn: a tag then holds one call.
    pub between_calls: Option<Box<str>>,
    /// What it writes after the last call (after its close, when it writes one), up to the end of
    /// the turn: a stop token that ends the turn is not written, since the stop decoder hides it.
    pub after_calls: Box<str>,
    /// How it writes a value that is not a string.
    pub values: Spellings,
    /// How it writes the thought, for a table with one.
    pub thought: Option<Message>,
    /// How it writes an answer.
    pub answer: Option<Message>,
}

/// Why a layout does not fit a table.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum LayoutError {
    /// The table's calls are not given as data, or their arguments are not tagged parameters,
    /// the one way of writing them a layout has places for.
    #[error("the table's calls are not tagged parameters given as data")]
    Calls,
    /// A text with places other than the ones its part holds once each.
    #[error("the {0} as written has places its part does not hold")]
    Places(&'static str),
    /// A part as written that its pattern does not read as the layout says: no match, a match that
    /// ends elsewhere, or a place the pattern does not capture.
    #[error("the {0} as written does not match its pattern where the layout says")]
    Unmatched(&'static str),
    /// Text around the parameters that holds the literal a parameter's tag begins with, where a
    /// search for parameters would find one.
    #[error("the text {0} the parameters holds the literal a parameter's tag begins with")]
    Trigger(&'static str),
    /// A terminal the layout names that has no such row in the table.
    #[error("the {0} is no terminal that closes it in the table")]
    Close(&'static str),
    /// A thought or an answer the table has no state for.
    #[error("the table has no state for the {0}")]
    Region(&'static str),
}

/// A sample of `min` characters (at least one) a class holds, for writing a place a pattern
/// captures: a letter where the class takes one.
fn sample(class: &Class, min: u16) -> String {
    let one = ['f', 'a', 'x', 'k', '0', '_', '-', '.']
        .into_iter()
        .chain(('\u{21}'..='\u{7e}').chain('\u{a1}'..='\u{2fff}'))
        .find(|&c| class.contains(c))
        .unwrap_or('f');
    std::iter::repeat_n(one, usize::from(min.max(1))).collect()
}

/// The class and the least length of the run `pattern` captures as `slot`.
fn captured(pattern: &Pattern, slot: Slot) -> Option<(&Class, u16)> {
    pattern.0.iter().find_map(|atom| match atom {
        Atom::Run {
            class,
            min,
            capture: Some(captured),
            ..
        } if *captured == slot => Some((class, *min)),
        _ => None,
    })
}

/// Whether `found` matched `text` up to `length`, each of `spans` where the layout put it.
fn fits(found: &Match, length: usize, spans: &[Option<(usize, usize)>; 3], slots: &[Slot]) -> bool {
    found.end() == length
        && slots.iter().all(|&slot| {
            let at = match slot {
                Slot::Name => 0,
                Slot::Key => 1,
                Slot::Value => 2,
            };
            found.span(slot).map(|span| (span.start, span.end)) == spans[at]
        })
}

impl Layout {
    /// Whether the layout fits `format`, a table whose calls are tagged parameters given as data
    /// ([`CallSyntax::Shaped`]): its texts hold the places their parts hold (the lead and the
    /// header the name, where the header captures it; a parameter its key and its value, once
    /// each), the header, a parameter and the trailer as written match the shape's patterns where
    /// the layout puts the places, with the text that follows each in a call; the text around the
    /// parameters holds no literal a parameter begins with; the call's close is a terminal with a
    /// row out of the call's state, or, for none, the table's stop row closes the call; and the
    /// table has a thought and an answer where the layout writes them, the stop ending the answer
    /// only where a stop token closes it, and a close the layout names for either a terminal with
    /// a row out of its state.
    pub fn check(&self, format: &Format) -> Result<(), LayoutError> {
        let Some(CallSyntax::Shaped(shape)) = format.call_syntax() else {
            return Err(LayoutError::Calls);
        };
        let Arguments::Params { tag, .. } = &shape.arguments else {
            return Err(LayoutError::Calls);
        };
        let call = &self.call;
        let header_names = matches!(shape.name, NameSource::Header);
        let names = |written: &Written| written.places(Slot::Key) + written.places(Slot::Value);
        if names(&call.lead) > 0 || call.lead.places(Slot::Name) > 1 {
            return Err(LayoutError::Places("lead"));
        }
        if names(&call.header) > 0 || call.header.places(Slot::Name) != usize::from(header_names) {
            return Err(LayoutError::Places("header"));
        }
        let parameter = &call.parameter;
        if parameter.places(Slot::Name) > 0
            || parameter.places(Slot::Key) != 1
            || parameter.places(Slot::Value) != 1
        {
            return Err(LayoutError::Places("parameter"));
        }
        let name = captured(&shape.header, Slot::Name)
            .map_or_else(|| "f".to_string(), |(class, min)| sample(class, min));
        let name = match &shape.name {
            NameSource::Literal(literal) => literal.to_string(),
            _ => name,
        };
        let key = captured(tag, Slot::Key).map_or_else(|| "k".to_string(), |(c, m)| sample(c, m));
        let terminal = call_terminal(format).unwrap_or_default();
        if self.header_for(shape, terminal, &name).is_none() {
            return Err(LayoutError::Unmatched("header"));
        }
        if self.parameter_for(shape, &key).is_none() {
            return Err(LayoutError::Unmatched("parameter"));
        }
        if shape.trailer.0.is_empty() != call.trailer.is_empty() {
            return Err(LayoutError::Unmatched("trailer"));
        }
        if !shape.trailer.0.is_empty() {
            let mut matcher = Matcher::new(&shape.trailer, &[], &shape.word, None).hold_cap(None);
            let trailer: &str = &call.trailer;
            if !reads(&mut matcher, trailer, trailer.len(), &[None; 3], &[]) {
                return Err(LayoutError::Unmatched("trailer"));
            }
        }
        if let Some(Atom::Lit(trigger)) = tag.0.first() {
            for (part, text) in [
                ("before", &call.before),
                ("between", &call.between),
                ("after", &call.after),
            ] {
                if text.contains(&**trigger) {
                    return Err(LayoutError::Trigger(part));
                }
            }
        }
        self.check_rows(format)
    }

    /// The header as written for a call to `name` after `terminal`, if the shape's header reads it
    /// back as a call to that tool, with the call's text after it as a stream reads it: the header
    /// captures exactly the name where the layout puts it, or, for a name the template writes,
    /// `name` is that one. `None` for a name the header's class does not hold, or one longer than
    /// the most a header holds undecided.
    pub(crate) fn header_for(
        &self,
        shape: &CallShape,
        terminal: &str,
        name: &str,
    ) -> Option<String> {
        let header_names = match &shape.name {
            NameSource::Header => true,
            NameSource::Literal(literal) if **literal == *name => false,
            NameSource::Literal(_) | NameSource::Body => return None,
        };
        let call = &self.call;
        let (header, spans) = call.header.filled(name, "", "");
        let follows = format!("{header}{}{}", call.before, self.sample_parameters(shape));
        let mut matcher = Matcher::new(
            &shape.header,
            &[],
            &shape.word,
            terminal.chars().next_back(),
        )
        .hold_cap(Some(shape.cap()));
        let slots: &[Slot] = if header_names { &[Slot::Name] } else { &[] };
        reads(&mut matcher, &follows, header.len(), &spans, slots).then_some(header)
    }

    /// One parameter as written for `key`, cut where its value stands, if the shape's tag reads it
    /// back with that key, its value where the layout puts it, with the call's text after it as a
    /// stream reads it. `None` for a key the tag's class does not hold.
    pub(crate) fn parameter_for(&self, shape: &CallShape, key: &str) -> Option<(String, String)> {
        let Arguments::Params { tag, ends, .. } = &shape.arguments else {
            return None;
        };
        let call = &self.call;
        let value =
            captured(tag, Slot::Value).map_or_else(|| "v".to_string(), |(c, m)| sample(c, m));
        let (written, spans) = call.parameter.filled("", key, &value);
        let rest = format!(
            "{written}{}{written}{}{}",
            call.between, call.after, call.trailer
        );
        // A tag begins with its trigger, a literal, so no `\b` reads the text before it.
        let mut matcher = Matcher::new(tag, ends, &shape.word, None).hold_cap(None);
        let slots = [Slot::Key, Slot::Value];
        reads(&mut matcher, &rest, written.len(), &spans, &slots)
            .then(|| call.parameter.around_the_value("", key))
    }

    /// Two parameters as written, with samples for the key and the value, and what follows the
    /// last: the call's text after the header, as a header's match reads on into it.
    fn sample_parameters(&self, shape: &CallShape) -> String {
        let call = &self.call;
        let (key, value) = match &shape.arguments {
            Arguments::Params { tag, .. } => (
                captured(tag, Slot::Key).map_or_else(|| "k".to_string(), |(c, m)| sample(c, m)),
                captured(tag, Slot::Value).map_or_else(|| "v".to_string(), |(c, m)| sample(c, m)),
            ),
            _ => ("k".to_string(), "v".to_string()),
        };
        let (written, _) = call.parameter.filled("", &key, &value);
        format!(
            "{written}{}{written}{}{}",
            call.between, call.after, call.trailer
        )
    }

    /// The rows the layout names: the call's close, the thought's and the answer's states, and
    /// the closes it names for them.
    fn check_rows(&self, format: &Format) -> Result<(), LayoutError> {
        let calls = (0..format.states()).find(|&s| format.emits(s) == Emits::Arguments);
        let Some(calls) = calls else {
            return Err(LayoutError::Calls);
        };
        let closes = |state: usize, text: &str| {
            format.transitions().any(|(source, on, target)| {
                source == state && target != state && format.terminal_text(on) == text
            })
        };
        match &self.call.close {
            Some(close) if !closes(calls, close) => return Err(LayoutError::Close("call's close")),
            None if !format.stop_closes(calls) => return Err(LayoutError::Close("call's stop")),
            _ => {}
        }
        if let Some(thought) = &self.thought {
            let reasoning = (0..format.states()).find(|&s| format.emits(s) == Emits::Reasoning);
            let Some(reasoning) = reasoning else {
                return Err(LayoutError::Region("thought"));
            };
            if let Some(close) = &thought.close {
                if !closes(reasoning, close) {
                    return Err(LayoutError::Close("thought's close"));
                }
            }
        }
        if let Some(answer) = &self.answer {
            let state = (0..format.states())
                .find(|&s| format.emits(s) == Emits::Content && !format.is_framing(s));
            let Some(state) = state else {
                return Err(LayoutError::Region("answer"));
            };
            let stopped = format.transitions().any(|(source, on, target)| {
                source == state && target != state && format.is_stop_token(format.terminal_text(on))
            });
            if answer.stops && state != 0 && !stopped {
                return Err(LayoutError::Close("answer's stop"));
            }
            if let Some(close) = &answer.close {
                if !closes(state, close) {
                    return Err(LayoutError::Close("answer's close"));
                }
            }
        }
        Ok(())
    }
}

/// The terminal of the first row into the table's call state.
fn call_terminal(format: &Format) -> Option<&str> {
    format
        .transitions()
        .find(|&(source, _, target)| source != target && format.emits(target) == Emits::Arguments)
        .map(|(_, on, _)| format.terminal_text(on))
}

/// Whether `matcher`, fed `text` to its end, matches it up to `length` with each of `slots` where
/// `spans` put it.
fn reads(
    matcher: &mut Matcher<'_>,
    text: &str,
    length: usize,
    spans: &[Option<(usize, usize)>; 3],
    slots: &[Slot],
) -> bool {
    let verdict = match matcher.feed(text) {
        Verdict::Pending => matcher.finish(),
        decided => decided,
    };
    matches!(verdict, Verdict::Matched(found) if fits(&found, length, spans, slots))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::{intern, CallShape, Dialect, End, ValueParser};

    fn none_of(chars: &str) -> Class {
        Class::new(chars.chars().map(|c| (c, c)), true)
    }

    fn run(class: Class, min: u16, greedy: bool, capture: Option<Slot>) -> Atom {
        Atom::Run {
            class,
            min,
            max: None,
            greedy,
            capture,
        }
    }

    fn lit(text: &str) -> Atom {
        Atom::Lit(text.into())
    }

    /// A shaped call on abstract tags: after `<c`, attributes skipped up to `n="NAME">`; each
    /// parameter `<p KEY>` and its value up to `</p>`; `trailer` before the close.
    fn shape(trailer: Pattern) -> CallShape {
        CallShape {
            header: Pattern(
                [
                    Atom::Boundary,
                    run(none_of(">"), 0, false, None),
                    Atom::Boundary,
                    lit("n=\""),
                    run(none_of("\""), 1, true, Some(Slot::Name)),
                    lit("\">"),
                ]
                .into(),
            ),
            name: NameSource::Header,
            arguments: Arguments::Params {
                tag: Pattern(
                    [
                        lit("<p "),
                        run(none_of(">"), 1, true, Some(Slot::Key)),
                        lit(">"),
                        run(Class::any(), 0, false, Some(Slot::Value)),
                    ]
                    .into(),
                ),
                ends: [End {
                    consume: Pattern([lit("</p>")].into()),
                    ahead: Pattern::default(),
                }]
                .into(),
                values: ValueParser::Raw,
            },
            trailer,
            word: Class::new([('0', '9'), ('A', 'Z'), ('_', '_'), ('a', 'z')], false),
            hold_cap: 8192,
        }
    }

    /// A table around `shape`: framing outside, a thought, an answer the stop token `<s>` ends,
    /// and the call between `<c` and `</c>`.
    fn table(shape: CallShape) -> Format {
        Format::new("laid out")
            .terminal("open", "<c")
            .terminal("close", "</c>")
            .terminal("think_open", "<t>")
            .terminal("think_close", "</t>")
            .terminal("answer_open", "<a>")
            .terminal("stop", "<s>")
            .state("outside", Emits::Content)
            .state("reasoning", Emits::Reasoning)
            .state("content", Emits::Content)
            .state("call", Emits::Arguments)
            .transition("outside", "open", "call")
            .transition("call", "close", "outside")
            .transition("outside", "think_open", "reasoning")
            .transition("reasoning", "think_close", "outside")
            .transition("outside", "answer_open", "content")
            .transition("content", "stop", "outside")
            .calls(CallSyntax::Shaped(intern(shape)))
            .framing("outside")
            .stop_tokens(["<s>"])
    }

    fn written(segments: impl IntoIterator<Item = Segment>) -> Written {
        Written(segments.into_iter().collect())
    }

    fn text(text: &str) -> Segment {
        Segment::Lit(text.into())
    }

    /// The layout the abstract table's template would render: `@NAME` and a newline before a
    /// call, a space before `n="NAME">`, one parameter a line.
    fn layout() -> Layout {
        Layout {
            call: CallLayout {
                lead: written([text("@"), Segment::Slot(Slot::Name), text("\n")]),
                header: written([text(" n=\""), Segment::Slot(Slot::Name), text("\">")]),
                before: "\n".into(),
                parameter: written([
                    text("<p "),
                    Segment::Slot(Slot::Key),
                    text(">"),
                    Segment::Slot(Slot::Value),
                    text("</p>"),
                ]),
                between: "\n".into(),
                after: "\n".into(),
                trailer: "".into(),
                close: Some("</c>".into()),
            },
            between_calls: Some("\n".into()),
            after_calls: "".into(),
            values: Spellings::default(),
            thought: Some(Message::default()),
            answer: Some(Message {
                stops: true,
                ..Message::default()
            }),
        }
    }

    #[test]
    fn a_layout_whose_parts_match_the_shape_fits() {
        let format = table(shape(Pattern::default()));
        assert_eq!(layout().check(&format), Ok(()));
        let with_layout = format.clone().with_layout(layout());
        assert_eq!(with_layout.layout(), Some(&layout()));
        assert_eq!(format.layout(), None);
        assert_eq!(with_layout.validate(), Ok(()), "parsing does not read it");
    }

    #[test]
    fn each_text_holds_the_places_its_part_holds() {
        let format = table(shape(Pattern::default()));
        let mut unnamed = layout();
        unnamed.call.header = Written::literal(" n=\"f\">");
        assert_eq!(unnamed.check(&format), Err(LayoutError::Places("header")));
        let mut keyed_lead = layout();
        keyed_lead.call.lead = written([Segment::Slot(Slot::Key)]);
        assert_eq!(keyed_lead.check(&format), Err(LayoutError::Places("lead")));
        let mut valueless = layout();
        valueless.call.parameter = written([text("<p "), Segment::Slot(Slot::Key), text(">")]);
        assert_eq!(
            valueless.check(&format),
            Err(LayoutError::Places("parameter"))
        );
        let mut literal = layout();
        literal.call.header = Written::literal(" n=\"f\">");
        let named = CallShape {
            name: NameSource::Literal("f".into()),
            ..shape(Pattern::default())
        };
        assert_eq!(
            literal.check(&table(named)),
            Ok(()),
            "a name the template writes has no place in the header"
        );
    }

    #[test]
    fn a_part_its_pattern_does_not_read_where_the_layout_says_does_not_fit() {
        let format = table(shape(Pattern::default()));
        let mut header = layout();
        header.call.header = written([text(" m=\""), Segment::Slot(Slot::Name), text("\">")]);
        assert_eq!(header.check(&format), Err(LayoutError::Unmatched("header")));
        let mut parameter = layout();
        parameter.call.parameter = written([
            text("<p "),
            Segment::Slot(Slot::Key),
            text(">"),
            Segment::Slot(Slot::Value),
            text("</q>"),
        ]);
        assert_eq!(
            parameter.check(&format),
            Err(LayoutError::Unmatched("parameter"))
        );
        // The key's place stands inside the value the pattern reads.
        let mut swapped = layout();
        swapped.call.parameter = written([
            text("<p "),
            Segment::Slot(Slot::Value),
            text(">"),
            Segment::Slot(Slot::Key),
            text("</p>"),
        ]);
        assert_eq!(
            swapped.check(&format),
            Err(LayoutError::Unmatched("parameter"))
        );
        let mut trigger = layout();
        trigger.call.between = "\n<p \n".into();
        assert_eq!(trigger.check(&format), Err(LayoutError::Trigger("between")));
    }

    #[test]
    fn a_trailer_as_written_matches_the_shapes() {
        let space = Class::new([(' ', ' '), ('\n', '\n')], false);
        let trailer = Pattern([lit("</f>"), run(space, 0, true, None)].into());
        let format = table(shape(trailer));
        let mut fits = layout();
        fits.call.trailer = "</f>\n".into();
        assert_eq!(fits.check(&format), Ok(()));
        assert_eq!(
            layout().check(&format),
            Err(LayoutError::Unmatched("trailer"))
        );
        let mut other = layout();
        other.call.trailer = "</g>".into();
        assert_eq!(other.check(&format), Err(LayoutError::Unmatched("trailer")));
        let mut surplus = layout();
        surplus.call.trailer = "</f>".into();
        assert_eq!(
            surplus.check(&table(shape(Pattern::default()))),
            Err(LayoutError::Unmatched("trailer")),
            "a shape without a trailer has none written"
        );
    }

    #[test]
    fn the_terminals_a_layout_names_close_what_it_says() {
        let format = table(shape(Pattern::default()));
        let mut other = layout();
        other.call.close = Some("<a>".into());
        assert_eq!(
            other.check(&format),
            Err(LayoutError::Close("call's close"))
        );
        let mut stopped = layout();
        stopped.call.close = None;
        assert_eq!(
            stopped.check(&format),
            Err(LayoutError::Close("call's stop"))
        );
        assert_eq!(
            stopped.check(&format.clone().closes_at_stop("call")),
            Ok(())
        );
        let mut answer = layout();
        answer.answer = Some(Message {
            stops: true,
            ..Message::default()
        });
        let unstopped = Format::new("laid out")
            .terminal("open", "<c")
            .terminal("close", "</c>")
            .terminal("answer_open", "<a>")
            .terminal("answer_close", "</a>")
            .state("outside", Emits::Content)
            .state("content", Emits::Content)
            .state("call", Emits::Arguments)
            .transition("outside", "open", "call")
            .transition("call", "close", "outside")
            .transition("outside", "answer_open", "content")
            .transition("content", "answer_close", "outside")
            .calls(CallSyntax::Shaped(intern(shape(Pattern::default()))))
            .framing("outside");
        answer.thought = None;
        assert_eq!(
            answer.check(&unstopped),
            Err(LayoutError::Close("answer's stop"))
        );
        assert_eq!(
            layout().check(&unstopped),
            Err(LayoutError::Region("thought"))
        );
        let mut thought = layout();
        thought.thought = Some(Message {
            close: Some("</t>".into()),
            ..Message::default()
        });
        assert_eq!(thought.check(&format), Ok(()));
        thought.thought = Some(Message {
            close: Some("<a>".into()),
            ..Message::default()
        });
        assert_eq!(
            thought.check(&format),
            Err(LayoutError::Close("thought's close"))
        );
        let mut answer = layout();
        answer.answer = Some(Message {
            close: Some("</t>".into()),
            ..Message::default()
        });
        assert_eq!(
            answer.check(&format),
            Err(LayoutError::Close("answer's close"))
        );
    }

    #[test]
    fn only_tagged_parameters_given_as_data_have_a_layout() {
        let json = CallShape {
            arguments: Arguments::Json(Dialect::default()),
            ..shape(Pattern::default())
        };
        assert_eq!(layout().check(&table(json)), Err(LayoutError::Calls));
        let plain = Format::new("plain")
            .terminal("open", "<c")
            .state("outside", Emits::Content)
            .state("call", Emits::Arguments)
            .transition("outside", "open", "call")
            .calls(CallSyntax::Json);
        assert_eq!(layout().check(&plain), Err(LayoutError::Calls));
    }
}
