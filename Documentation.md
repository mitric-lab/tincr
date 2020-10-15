# Notes on code documentation

In order to have a uniform documentation of the code, it is useful to define a
few basic principles of documentation. I took these hints from the official
[Rust repository](https://github.com/rust-lang/rfcs/blob/master/text/1574-more-api-documentation-conventions.md#appendix-a-full-conventions-text). 

Below is a combination of RFC 505 + this RFC’s modifications, for convenience.

### Summary sentence
[summary-sentence]: #summary-sentence

In API documentation, the first line should be a single-line short sentence
providing a summary of the code. This line is used as a summary description
throughout Rustdoc’s output, so it’s a good idea to keep it short.

The summary line should be written in third person singular present indicative
form. Basically, this means write ‘Returns’ instead of ‘Return’.

### English
[english]: #english

This section applies to `rustc` and the standard library.

All documentation for the standard library is standardized on American English,
with regards to spelling, grammar, and punctuation conventions. Language
changes over time, so this doesn’t mean that there is always a correct answer
to every grammar question, but there is often some kind of formal consensus.

### Use line comments
[use-line-comments]: #use-line-comments

Avoid block comments. Use line comments instead:

```rust
// Wait for the main task to return, and set the process error code
// appropriately.
```

Instead of:

```rust
/*
 * Wait for the main task to return, and set the process error code
 * appropriately.
 */
```

Only use inner doc comments `//!` to write crate and module-level documentation,
nothing else. When using `mod` blocks, prefer `///` outside of the block:

```rust
/// This module contains tests
mod test {
    // ...
}
```

over

```rust
mod test {
    //! This module contains tests

    // ...
}
```

### Using Markdown
[using-markdown]: #using-markdown

Within doc comments, use Markdown to format your documentation.

Use top level headings (`#`) to indicate sections within your comment. Common headings:

* Examples
* Panics
* Errors
* Safety
* Aborts
* Undefined Behavior

An example:

```rust
/// # Examples
```

Even if you only include one example, use the plural form: ‘Examples’ rather
than ‘Example’. Future tooling is easier this way.

Use backticks (`) to denote a code fragment within a sentence.

Use triple backticks (```) to write longer examples, like this:

    This code does something cool.

    ```rust
    let x = foo();

    x.bar();
    ```

When appropriate, make use of Rustdoc’s modifiers. Annotate triple backtick blocks with
the appropriate formatting directive.

    ```rust
    println!("Hello, world!");
    ```

    ```ruby
    puts "Hello"
    ```

In API documentation, feel free to rely on the default being ‘rust’:

    /// For example:
    ///
    /// ```
    /// let x = 5;
    /// ```

In long-form documentation, always be explicit:

    For example:

    ```rust
    let x = 5;
    ```

This will highlight syntax in places that do not default to ‘rust’, like GitHub.

Rustdoc is able to test all Rust examples embedded inside of documentation, so
it’s important to mark what is not Rust so your tests don’t fail.

References and citation should be linked ‘reference style.’ Prefer

```
[Rust website]

[Rust website]: http://www.rust-lang.org
```

to

```
[Rust website](http://www.rust-lang.org)
```

If the text is very long, feel free to use the shortened form:

```
This link [is very long and links to the Rust website][website].

[website]: http://www.rust-lang.org
```
