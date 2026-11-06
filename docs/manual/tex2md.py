"""One of the manuals' LaTeX sources as GitHub Markdown.

    python tex2md.py SOURCE.tex AUX OUT.md [--figdir manual/fig]

There is no pandoc here, and the manuals are written in a small subset of LaTeX, so this
converts exactly that subset: sections (numbered as LaTeX numbers them), paragraphs, lists,
display math (numbered from the aux file), the preamble's boxes, figures (each PDF's PNG
twin under `--figdir`), tables, algorithms (as indented text), code listings, the
bibliography, and inline markup. Cross-references and citations are resolved from the aux
file of the LaTeX build, so the Markdown numbers every section, equation, figure, table and
algorithm as the PDF does, and links the sections.

Math is passed through as written; the manuals use no custom math macros, so GitHub's
MathJax renders it as LaTeX does. A command this does not know is reported on stderr and
dropped with its arguments' text kept, so a new construct shows up rather than vanishing.
"""
import argparse
import re
import sys
from collections import OrderedDict
from pathlib import Path

WARN = []


def warn(msg):
    WARN.append(msg)


#######################################################################################
# Scanning
#######################################################################################


def match_brace(s, i):
    """Index just past the group that opens at `s[i] == "{"` (or "[")."""
    open_ch = s[i]
    close_ch = {"{": "}", "[": "]"}[open_ch]
    depth = 0
    j = i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    raise ValueError(f"unbalanced {open_ch} at {s[i:i + 60]!r}")


def skip_ws(s, i):
    while i < len(s) and s[i] in " \t\n":
        i += 1
    return i


def read_args(s, i, nopt, nreq):
    """Optional [..] args (at most `nopt`) then `nreq` {..} args from `s[i:]`."""
    opts, reqs = [], []
    j = i
    for _ in range(nopt):
        k = skip_ws(s, j) if nreq else j
        if k < len(s) and s[k] == "[":
            e = match_brace(s, k)
            opts.append(s[k + 1:e - 1])
            j = e
    for _ in range(nreq):
        k = skip_ws(s, j)
        if k >= len(s) or s[k] != "{":
            raise ValueError(f"expected an argument at {s[k:k + 60]!r}")
        e = match_brace(s, k)
        reqs.append(s[k + 1:e - 1])
        j = e
    return opts, reqs, j


def find_end(s, i, env):
    """Index of `\\end{env}` matching an environment whose body starts at `i`."""
    depth = 1
    pat = re.compile(r"\\(begin|end)\{" + re.escape(env) + r"\}")
    for m in pat.finditer(s, i):
        depth += 1 if m.group(1) == "begin" else -1
        if depth == 0:
            return m.start(), m.end()
    raise ValueError(f"no \\end{{{env}}}")


def split_top(s, sep):
    """`s` split at `sep` outside braces, math and escapes (`\\&` is not a separator)."""
    out, depth, cur, i, math = [], 0, [], 0, False
    while i < len(s):
        c = s[i]
        if c == "\\":
            if s.startswith(sep, i) and depth == 0 and not math:
                out.append("".join(cur))
                cur = []
                i += len(sep)
                continue
            cur.append(s[i:i + 2])
            i += 2
            continue
        if c == "$":
            math = not math
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        if s.startswith(sep, i) and depth == 0 and not math:
            out.append("".join(cur))
            cur = []
            i += len(sep)
            continue
        cur.append(c)
        i += 1
    out.append("".join(cur))
    return out


def strip_comments(s):
    out = []
    for line in s.split("\n"):
        m = re.search(r"(?<!\\)%", line)
        if m:
            line = line[:m.start()]
            if not line.strip():
                continue  # a whole-line comment leaves no blank line behind
        out.append(line)
    return "\n".join(out)


#######################################################################################
# The aux file: numbers and kinds of labels
#######################################################################################


def read_aux(path):
    """`{label: (kind, number)}` from LaTeX's aux, kinds as cleveref records them."""
    labels = {}
    kinds = {}
    try:
        text = open(path).read()
    except FileNotFoundError:
        warn(f"no aux file {path}: references unresolved")
        return labels
    for m in re.finditer(r"\\newlabel\{([^}]*)\}\{\{(.*)$", text, re.M):
        key, rest = m.group(1), m.group(2)
        if key.endswith("@cref"):
            km = re.match(r"\[([^\]]*)\]\[[^\]]*\]\[[^\]]*\]([^}]*)\}", rest)
            if km:
                kinds[key[:-5]] = (km.group(1), km.group(2))
        else:
            nm = re.match(r"([^}]*)\}", rest)
            if nm:
                labels.setdefault(key, (None, nm.group(1)))
    for key, (kind, num) in kinds.items():
        labels[key] = (kind, num)
    return labels


#######################################################################################
# The converter
#######################################################################################

NAMES = {
    "section": ("Section", "Sections"),
    "subsection": ("Section", "Sections"),
    "subsubsection": ("Section", "Sections"),
    "appendix": ("Appendix", "Appendices"),
    "subappendix": ("Appendix", "Appendices"),
    "subsubappendix": ("Appendix", "Appendices"),
    "equation": ("Eq.", "Eqs."),
    "figure": ("Figure", "Figures"),
    "table": ("Table", "Tables"),
    "algorithm": ("Algorithm", "Algorithms"),
}

ACCENTS = {
    "'": {"e": "é", "a": "á", "o": "ó", "i": "í", "u": "ú", "E": "É"},
    "`": {"e": "è", "a": "à"},
    '"': {"o": "ö", "a": "ä", "u": "ü", "O": "Ö", "A": "Ä", "U": "Ü"},
    "^": {"e": "ê", "o": "ô"},
    "~": {"n": "ñ"},
}

# Declarations and spacing that leave nothing behind in Markdown.
DROP0 = {
    "small", "footnotesize", "scriptsize", "normalsize", "large", "Large", "huge", "noindent",
    "centering", "raggedright", "sffamily", "bfseries", "itshape", "rmfamily", "ttfamily",
    "medskip", "bigskip", "smallskip", "newpage", "clearpage", "par", "protect", "hfill",
    "quad", "qquad", "relax", "nobreak", "allowbreak", "linebreak", "break", "maketitle",
    "toprule", "midrule", "bottomrule", "hline", "appendix", "selectfont", "strut",
    "begingroup", "endgroup",
}
DROP1 = {"label", "vspace", "hspace", "vspace*", "hspace*", "phantom", "pageref", "needspace", "Needspace"}
DROP2 = {"setcounter", "addtocounter", "setlength"}
KEEP1 = {"mbox", "textrm", "textsf", "textsc", "textup", "textnormal", "text", "hbox", "makebox"}
SYMBOLS = {
    "ldots": "…", "dots": "…", "textellipsis": "…", "S": "§", "textmu": "µ", "micro": "µ",
    "textdegree": "°", "textbackslash": "\\", "textasciitilde": "~", "textless": "<",
    "textgreater": ">", "textbar": "|", "textendash": "–", "textemdash": "—", "LaTeX": "LaTeX",
    "TeX": "TeX", "textbullet": "•", "copyright": "©", "texttimes": "×", "ss": "ß",
    "textquotesingle": "'", "checkmark": "✓",
}


class Converter:
    def __init__(self, src, labels, figdir):
        self.labels = labels
        self.figdir = figdir.rstrip("/")
        self.cites = OrderedDict()
        self.sec = [0, 0, 0]
        self.appendix = False
        self.anchors = {}  # label -> markdown anchor of its heading
        self.slugs = {}
        self.headings = []  # (level, number, text, anchor) for a table of contents
        self.protected = []
        body = strip_comments(src)
        m = re.search(r"\\begin\{document\}(.*)\\end\{document\}", body, re.S)
        self.body = m.group(1) if m else body
        self._prescan()

    # -- before converting: bibliography order, heading numbers and anchors --------

    def _prescan(self):
        for m in re.finditer(r"\\bibitem\{([^}]*)\}", self.body):
            self.cites[m.group(1)] = len(self.cites) + 1
        sec = [0, 0, 0]
        appendix = False
        pat = re.compile(r"\\(appendix)\b|\\(section|subsection|subsubsection)(\*?)\s*\{")
        for m in pat.finditer(self.body):
            if m.group(1):
                appendix = True
                sec = [0, 0, 0]
                continue
            level = ["section", "subsection", "subsubsection"].index(m.group(2))
            star = m.group(3) == "*"
            _, (title,), end = read_args(self.body, m.end() - 1, 0, 1)
            num = None
            if not star:
                sec[level] += 1
                for k in range(level + 1, 3):
                    sec[k] = 0
                parts = [chr(ord("A") + sec[0] - 1) if appendix else str(sec[0])] + [str(x) for x in sec[1:level + 1]]
                num = ".".join(parts)
            text = self.inline(title)
            anchor = self._slug(f"{num} {text}" if num else text)
            lm = re.match(r"\s*\\label\{([^}]*)\}", self.body[end:])
            if lm:
                self.anchors[lm.group(1)] = anchor
                kind, lnum = self.labels.get(lm.group(1), (None, None))
                if lnum is not None and num is not None and lnum != num:
                    warn(f"heading {title!r}: numbered {num} here, {lnum} in the aux")
            self.headings.append((level, num, text, anchor))
        self.labels_used = set()

    def _slug(self, text):
        t = re.sub(r"[`*]", "", text).lower()
        t = re.sub(r"[^\w\- ]", "", t)
        t = t.replace(" ", "-")
        n = self.slugs.get(t, 0)
        self.slugs[t] = n + 1
        return t if n == 0 else f"{t}-{n}"

    # -- references ------------------------------------------------------------------

    def ref_text(self, keys, cap):
        groups = OrderedDict()
        for key in keys:
            key = key.strip()
            kind, num = self.labels.get(key, (None, None))
            if num is None:
                warn(f"unresolved reference {key}")
                kind, num = "unknown", "??"
            groups.setdefault(kind, []).append((key, num))
        parts = []
        for kind, items in groups.items():
            one, many = NAMES.get(kind, ("", ""))
            nums = []
            for key, num in items:
                txt = f"({num})" if kind == "equation" else num
                if key in self.anchors:
                    txt = f"[{txt}](#{self.anchors[key]})"
                nums.append(txt)
            name = one if len(nums) == 1 else many
            if len(nums) == 1 and items[0][0] in self.anchors:
                parts.append(f"[{name} {items[0][1]}](#{self.anchors[items[0][0]]})")
                continue
            if len(nums) == 1:
                joined = nums[0]
            elif len(nums) == 2:
                joined = f"{nums[0]} and {nums[1]}"
            else:
                joined = ", ".join(nums[:-1]) + f" and {nums[-1]}"
            parts.append(f"{name} {joined}".strip())
        return " and ".join(parts)

    # -- inline markup ---------------------------------------------------------------

    def protect(self, text):
        self.protected.append(text)
        return f"\x00{len(self.protected) - 1}\x00"

    def restore(self, s):
        while "\x00" in s:
            s = re.sub(r"\x00(\d+)\x00", lambda m: self.protected[int(m.group(1))], s)
        return s

    def inline(self, s, code=False):
        """Inline LaTeX as Markdown. `code`: inside a code block, so no Markdown markup."""
        out = []
        i = 0
        n = len(s)
        while i < n:
            c = s[i]
            if c == "$":
                if s.startswith("$$", i):
                    j = s.index("$$", i + 2)
                    out.append(self.protect(s[i:j + 2]))
                    i = j + 2
                    continue
                j = i + 1
                while j < n and not (s[j] == "$" and s[j - 1] != "\\"):
                    j += 1
                math = s[i:j + 1]
                out.append(self.protect(self.clean_math(math)))
                i = j + 1
                continue
            if c == "\\":
                if i + 1 < n and not s[i + 1].isalpha():
                    ch = s[i + 1]
                    if ch in ACCENTS:
                        k = i + 2
                        if k < n and s[k] == "{":
                            e = match_brace(s, k)
                            letter = s[k + 1:e - 1]
                            i = e
                        else:
                            letter = s[k]
                            i = k + 1
                        out.append(ACCENTS[ch].get(letter, letter))
                        continue
                    if ch == "\\":
                        out.append(" ")
                        i += 2
                        if i < n and s[i] == "[":
                            i = match_brace(s, i)
                        continue
                    if ch in ",;:! ":
                        out.append(" ")
                        i += 2
                        continue
                    if ch == "$":
                        out.append(self.protect("\\$"))
                        i += 2
                        continue
                    out.append({"%": "%", "&": "&", "#": "#", "_": "_", "{": "{", "}": "}", "-": ""}.get(ch, ch))
                    i += 2
                    continue
                m = re.match(r"\\([A-Za-z]+\*?)", s[i:])
                name = m.group(1)
                j = i + len(m.group(0))
                res, j = self.command(name, s, j, code)
                out.append(res)
                i = j
                continue
            if c in "{}":
                i += 1
                continue
            if c == "~":
                out.append(" ")
                i += 1
                continue
            out.append(c)
            i += 1
        text = "".join(out)
        if not code:
            text = text.replace("---", "—").replace("--", "–").replace("``", "“").replace("''", "”")
        text = re.sub(r"[ \t]+", " ", text)
        return self.restore(text) if not code else self.restore(text)

    def clean_math(self, m):
        m = re.sub(r"\\label\{[^}]*\}", "", m)
        return m

    def command(self, name, s, j, code):
        """One command at `s[j:]` (just past its name). Returns (markdown, new index)."""
        def args(nopt, nreq):
            o, r, e = read_args(s, j, nopt, nreq)
            return o, r, e

        if name in ("textbf",):
            _, (a,), e = args(0, 1)
            t = self.inline(a, code)
            return (t if code else f"**{t.strip()}**"), e
        if name in ("emph", "textit", "textsl"):
            _, (a,), e = args(0, 1)
            t = self.inline(a, code)
            return (t if code else f"*{t.strip()}*"), e
        if name in ("code", "texttt", "path"):
            _, (a,), e = args(0, 1)
            raw = a.replace("\\_", "_").replace("\\textbackslash", "\\").replace("\\{", "{").replace("\\}", "}")
            raw = raw.replace("\\&", "&").replace("\\%", "%").replace("\\#", "#").replace("\\$", "$")
            if name == "texttt" and "\\" in raw:
                raw = self.inline(a, True)
            return (raw if code else self.protect(f"`{raw}`")), e
        if name == "url":
            _, (a,), e = args(0, 1)
            return self.protect(f"<{a}>"), e
        if name == "href":
            _, (u, t), e = args(0, 2)
            return self.protect(f"[{self.inline(t)}]({u})"), e
        if name in ("cref", "Cref"):
            _, (a,), e = args(0, 1)
            return self.protect(self.ref_text(a.split(","), name == "Cref")), e
        if name == "eqref":
            _, (a,), e = args(0, 1)
            kind, num = self.labels.get(a, (None, "??"))
            if num == "??":
                warn(f"unresolved reference {a}")
            return f"({num})", e
        if name == "ref":
            _, (a,), e = args(0, 1)
            kind, num = self.labels.get(a, (None, "??"))
            if num == "??":
                warn(f"unresolved reference {a}")
            if a in self.anchors:
                return self.protect(f"[{num}](#{self.anchors[a]})"), e
            return num, e
        if name == "cite":
            o, (a,), e = args(1, 1)
            nums = []
            for k in a.split(","):
                k = k.strip()
                if k not in self.cites:
                    warn(f"unknown citation {k}")
                nums.append(str(self.cites.get(k, "?")))
            extra = f", {self.inline(o[0])}" if o else ""
            return f"[{', '.join(nums)}{extra}]", e
        if name == "paragraph":
            _, (a,), e = args(0, 1)
            t = self.inline(a)
            return (f"**{t.strip()}.** " if not t.strip().endswith((".", "?", ":")) else f"**{t.strip()}** "), e
        if name == "footnote":
            _, (a,), e = args(0, 1)
            return f" ({self.inline(a, code).strip()})", e
        if name == "textcolor":
            _, (_c, a), e = args(0, 2)
            return self.inline(a, code), e
        if name == "colorbox":
            _, (_c, a), e = args(0, 2)
            return self.inline(a, code), e
        if name == "texorpdfstring":
            _, (a, b), e = args(0, 2)
            return self.inline(b, code), e
        if name in KEEP1:
            _, (a,), e = args(0, 1)
            return self.inline(a, code), e
        if name in DROP2:
            _, _, e = args(0, 2)
            return "", e
        if name in DROP1:
            o, (a,), e = args(1, 1)
            return "", e
        if name in DROP0:
            return "", j
        if name in SYMBOLS:
            e = j
            if s.startswith("{}", e):
                e += 2
            return SYMBOLS[name], e
        if name in ("color",):
            _, (a,), e = args(0, 1)
            return "", e
        if name == "rule":
            o, (a, b), e = args(1, 2)
            return "", e
        if name in ("times", "pm", "approx", "to", "le", "ge", "rightarrow", "leftarrow"):
            return {"times": "×", "pm": "±", "approx": "≈", "to": "→", "le": "≤", "ge": "≥",
                    "rightarrow": "→", "leftarrow": "←"}[name], j
        if name == "includegraphics":
            o, (a,), e = args(1, 1)
            return self.image(a), e
        warn(f"unknown command \\{name}")
        return "", j

    def image(self, path, alt=""):
        stem = re.sub(r"\.(pdf|png|jpg)$", "", path.split("/")[-1])
        return self.protect(f"![{alt}]({self.figdir}/{stem}.png)")

    # -- blocks ----------------------------------------------------------------------

    def blocks(self, s):
        """Markdown lines for a run of LaTeX blocks."""
        lines = []
        i = 0
        n = len(s)
        text_start = 0
        block_pat = re.compile(
            r"\\begin\{([A-Za-z*]+)\}|\\(section|subsection|subsubsection)(\*?)\s*\{|\\(manualtitle|tableofcontents|appendix)\b"
            r"|(\\\[)")
        while True:
            m = block_pat.search(s, i)
            if not m:
                lines += self.paragraphs(s[text_start:])
                break
            lines += self.paragraphs(s[text_start:m.start()])
            if m.group(5):
                e = s.index("\\]", m.end())
                lines += ["$$", s[m.end():e].strip(), "$$", ""]
                i = text_start = e + 2
            elif m.group(1):
                env = m.group(1)
                body_start = m.end()
                b_end, e_end = find_end(s, body_start, env)
                lines += self.environment(env, s[body_start:b_end])
                i = text_start = e_end
            elif m.group(2):
                level = ["section", "subsection", "subsubsection"].index(m.group(2))
                star = m.group(3) == "*"
                _, (title,), e = read_args(s, m.end() - 1, 0, 1)
                lines += self.heading(level, star, title)
                i = text_start = e
            else:
                cmd = m.group(4)
                e = m.end()
                if cmd == "manualtitle":
                    _, (t, sub, meta), e = read_args(s, e, 0, 3)
                    lines += [f"# {self.inline(t).strip()}", "", f"*{self.inline(sub).strip()}*", "",
                              self.inline(meta).strip(), ""]
                elif cmd == "tableofcontents":
                    lines += self.toc()
                elif cmd == "appendix":
                    self.appendix = True
                    self.sec = [0, 0, 0]
                i = text_start = e
        return lines

    def toc(self):
        out = ["**Contents**", ""]
        for level, num, text, anchor in self.headings:
            if level > 1:
                continue
            ind = "  " * level
            label = f"{num} {text}" if num else text
            out.append(f"{ind}- [{label}](#{anchor})")
        return out + [""]

    def heading(self, level, star, title):
        num = None
        if not star:
            self.sec[level] += 1
            for k in range(level + 1, 3):
                self.sec[k] = 0
            parts = [chr(ord("A") + self.sec[0] - 1) if self.appendix else str(self.sec[0])]
            parts += [str(x) for x in self.sec[1:level + 1]]
            num = ".".join(parts)
        text = self.inline(title).strip()
        hashes = "#" * (level + 2)
        return [f"{hashes} {num} {text}" if num else f"{hashes} {text}", ""]

    def paragraphs(self, s):
        out = []
        for para in re.split(r"\n\s*\n", s):
            t = self.inline(para).strip()
            t = re.sub(r"\s*\n\s*", " ", t)
            if t:
                out += [t, ""]
        return out

    def environment(self, env, body):
        star = env.endswith("*")
        base = env.rstrip("*")
        if base in ("itemize", "enumerate", "description"):
            return self.listing(base, body)
        if base in ("equation", "align", "gather", "multline", "displaymath"):
            return self.display(base, star, body)
        if base == "equationbox":
            _, (title,), e = read_args(body, 0, 0, 1)
            inner = self.blocks(body[e:])
            return self.quote([f"**{self.inline(title).strip()}**", ""] + inner)
        if base == "codebox":
            return self.quote(self.blocks(body))
        if base == "notebox":
            _, (title,), e = read_args(body, 0, 0, 1)
            inner = self.blocks(body[e:])
            t = self.inline(title).strip()
            if inner:
                inner[0] = f"**{t}.** {inner[0]}"
            return self.quote(inner)
        if base == "figure":
            return self.figure(body)
        if base == "table":
            return self.table_float(body)
        if base in ("longtable", "xltabular") and "\\caption" in body:
            # A long table carries its caption and label in its first row: lift them out,
            # numbered as a table float's.
            nreq = {"longtable": 1, "xltabular": 2}[base]
            _, _, e = read_args(body, 0, 1, nreq)
            spec, rest = body[:e], body[e:]
            m = re.search(r"\\caption\s*\{", rest)
            end = match_brace(rest, m.end() - 1)
            tail = rest[end:]
            lm = re.match(r"\s*\\label\{([^}]*)\}", tail)
            cut = end + (lm.end() if lm else 0)
            rows = rest[cut:]
            rows = re.sub(r"^\s*\\\\", "", rows)
            cap, label = self.caption_label(rest[:cut])
            kind, num = self.labels.get(label, (None, "?")) if label else (None, "")
            # The rows a long table repeats on every page (up to `\endhead`) are its header
            # once; the first page's (`\endfirsthead`) are the ones kept.
            if "\\endfirsthead" in rows:
                first, _, after = rows.partition("\\endfirsthead")
                rows = first + after.partition("\\endhead")[2]
            return [f"**Table {num}: {self.inline(cap).strip()}**", ""] + self.table(base, spec + rows)
        if base in ("tabular", "tabularx", "longtable", "xltabular"):
            return self.table(base, body)
        if base == "algorithm":
            return self.algorithm(body)
        if base == "algorithmic":
            return ["```text"] + self.algorithmic(body) + ["```", ""]
        if base == "lstlisting":
            lang = "python"
            m = re.match(r"\[([^\]]*)\]", body)
            if m:
                lm = re.search(r"language=(\w+)", m.group(1))
                lang = lm.group(1).lower() if lm else ""
                if "language=" not in m.group(1):
                    lang = "python"
                body = body[m.end():]
            if lang in ("bash", "sh"):
                lang = "bash"
            return [f"```{lang}"] + body.strip("\n").split("\n") + ["```", ""]
        if base == "verbatim":
            return ["```text"] + body.strip("\n").split("\n") + ["```", ""]
        if base in ("center", "flushleft", "flushright", "minipage", "small", "footnotesize", "adjustbox", "sloppypar"):
            if base in ("minipage", "adjustbox"):
                _, _, e = read_args(body, 0, 1, 1)
                body = body[e:]
            return self.blocks(body)
        if base == "quote":
            return self.quote(self.blocks(body))
        if base == "thebibliography":
            return self.bibliography(body)
        warn(f"unknown environment {env}")
        return self.blocks(body)

    def quote(self, lines):
        out = []
        for line in (sub for line in lines for sub in line.split("\n")):
            out.append(f"> {line}" if line else ">")
        while out and out[-1] == ">":
            out.pop()
        return out + [""]

    def listing(self, kind, body):
        items = []
        depth = 0
        cur = None
        i = 0
        pat = re.compile(r"\\item\b|\\begin\{(itemize|enumerate|description)\}|\\end\{(itemize|enumerate|description)\}")
        starts = []
        for m in pat.finditer(body):
            if m.group(1):
                depth += 1
            elif m.group(2):
                depth -= 1
            elif depth == 0:
                starts.append(m.start())
        starts.append(len(body))
        out = []
        for k in range(len(starts) - 1):
            chunk = body[starts[k] + len("\\item"):starts[k + 1]]
            label = None
            if kind == "description" or chunk.lstrip().startswith("["):
                c = chunk.lstrip()
                if c.startswith("["):
                    e = match_brace(c, 0)
                    label = c[1:e - 1]
                    chunk = c[e:]
            inner = self.blocks(chunk)
            while inner and inner[-1] == "":
                inner.pop()
            marker = f"{k + 1}." if kind == "enumerate" else "-"
            pad = " " * (len(marker) + 1)
            if label is not None:
                lab = self.inline(label).strip()
                first = f"**{lab}**" if not lab.startswith("**") else lab
                inner = [f"{first} {inner[0]}" if inner else first] + inner[1:]
            if not inner:
                inner = [""]
            out.append(f"{marker} {inner[0]}")
            for line in inner[1:]:
                out.append(f"{pad}{line}" if line else "")
        return out + [""]

    def display(self, base, star, body):
        labels = re.findall(r"\\label\{([^}]*)\}", body)
        tex = re.sub(r"\\label\{[^}]*\}", "", body)
        tex = re.sub(r"\\(nonumber|notag)\b", "", tex).strip()
        if base in ("align", "gather"):
            env = "aligned" if base == "align" else "gathered"
            tex = f"\\begin{{{env}}}\n{tex}\n\\end{{{env}}}"
        if base == "multline":
            tex = f"\\begin{{split}}\n{tex}\n\\end{{split}}"
        tag = ""
        if labels:
            kind, num = self.labels.get(labels[0], (None, None))
            if num is None:
                warn(f"unresolved equation label {labels[0]}")
            else:
                tag = f" \\tag{{{num}}}"
            if len(labels) > 1:
                warn(f"several labels in one display: {labels}")
        return ["$$", tex + tag, "$$", ""]

    def caption_label(self, body):
        cap, label = "", None
        m = re.search(r"\\caption\s*(\[)?", body)
        if m:
            k = m.end() - 1 if m.group(1) else m.end()
            o, (cap,), _ = read_args(body, m.start() + len("\\caption"), 1, 1)
        lm = re.search(r"\\label\{([^}]*)\}", body)
        if lm:
            label = lm.group(1)
        return cap, label

    def figure(self, body):
        cap, label = self.caption_label(body)
        kind, num = self.labels.get(label, (None, "?")) if label else (None, "")
        imgs = [m.group(2) for m in re.finditer(r"\\includegraphics\s*(\[[^\]]*\])?\s*\{([^}]*)\}", body)]
        out = []
        for path in imgs:
            stem = re.sub(r"\.(pdf|png|jpg)$", "", path.split("/")[-1])
            out.append(f"![Figure {num}]({self.figdir}/{stem}.png)")
        out.append("")
        if cap:
            out += [f"*Figure {num}: {self.inline(cap).strip()}*", ""]
        return out

    def table_float(self, body):
        cap, label = self.caption_label(body)
        kind, num = self.labels.get(label, (None, "?")) if label else (None, "")
        out = []
        if cap:
            out += [f"**Table {num}: {self.inline(cap).strip()}**", ""]
        m = re.search(r"\\begin\{(tabular|tabularx|longtable|xltabular)\}", body)
        if m:
            b_end, _ = find_end(body, m.end(), m.group(1))
            out += self.table(m.group(1), body[m.end():b_end])
        else:
            warn("a table float without a tabular")
        return out

    def table(self, kind, body):
        nreq = {"tabular": 1, "tabularx": 2, "longtable": 1, "xltabular": 2}[kind]
        _, _, e = read_args(body, 0, 1, nreq)
        body = body[e:]
        body = re.sub(r"\\(toprule|midrule|bottomrule|hline|endhead|endfirsthead|endfoot|endlastfoot|addlinespace)\b(\[[^\]]*\])?", "", body)
        body = re.sub(r"\\cmidrule(\([^)]*\))?\{[^}]*\}", "", body)
        body = re.sub(r"\\rowcolor(\[[^\]]*\])?\{[^}]*\}", "", body)
        rows = [r for r in split_top(body, "\\\\") if r.strip()]
        table = []
        for r in rows:
            r = re.sub(r"^\s*\[[^\]]*\]", "", r)  # `\\[2pt]` leaves its spacing on the next row
            cells = []
            for cell in split_top(r, "&"):
                mc = re.match(r"\s*\\multicolumn\{(\d+)\}\{[^}]*\}\{", cell)
                span = 1
                if mc:
                    span = int(mc.group(1))
                    e = match_brace(cell, mc.end() - 1)
                    cell = cell[mc.end():e - 1]
                t = self.inline(cell).strip()
                t = re.sub(r"\s*\n\s*", " ", t)
                if "|" in t.replace("\\|", ""):
                    warn(f"a '|' in a table cell: {t[:60]!r}")
                    t = t.replace("|", "\\|")
                cells.append(t)
                cells += [""] * (span - 1)
            table.append(cells)
        if not table:
            return []
        width = max(len(r) for r in table)
        out = []
        for k, r in enumerate(table):
            r = r + [""] * (width - len(r))
            out.append("| " + " | ".join(r) + " |")
            if k == 0:
                out.append("|" + "---|" * width)
        return out + [""]

    def algorithm(self, body):
        cap, label = self.caption_label(body)
        kind, num = self.labels.get(label, (None, "?")) if label else (None, "")
        out = []
        if cap:
            out += [f"**Algorithm {num}: {self.inline(cap).strip()}**", ""]
        m = re.search(r"\\begin\{algorithmic\}(\[[^\]]*\])?", body)
        if m:
            b_end, _ = find_end(body, m.end(), "algorithmic")
            out += ["```text"] + self.algorithmic(body[m.end():b_end]) + ["```", ""]
        return out

    def algorithmic(self, body):
        toks = re.compile(
            r"\\(State|Statex|If|ElsIf|Else|EndIf|For|ForAll|EndFor|While|EndWhile|Repeat|Until|Loop|"
            r"EndLoop|Function|EndFunction|Procedure|EndProcedure|Require|Ensure|Return|Comment)\b")
        pieces = []
        last = 0
        cur = None
        for m in toks.finditer(body):
            if cur is not None:
                pieces.append((cur, body[last:m.start()]))
            cur = m.group(1)
            last = m.end()
        if cur is not None:
            pieces.append((cur, body[last:]))
        lines = []
        ind = 0

        def emit(text):
            lines.append("    " * ind + text)

        for name, rest in pieces:
            def arg(k=1):
                _, r, e = read_args(rest, 0, 0, k)
                return r, rest[e:]

            if name in ("State", "Statex"):
                emit(self.inline(rest, True).strip())
            elif name == "Comment":
                (c,), tail = arg()
                if lines:
                    lines[-1] = lines[-1] + "    # " + self.inline(c, True).strip()
                else:
                    emit("# " + self.inline(c, True).strip())
                if tail.strip():
                    emit(self.inline(tail, True).strip())
            elif name in ("If", "ElsIf", "For", "ForAll", "While", "Until"):
                (c,), tail = arg()
                word = {"If": "if", "ElsIf": "else if", "For": "for", "ForAll": "for all",
                        "While": "while", "Until": "until"}[name]
                end = {"If": " then", "ElsIf": " then", "For": " do", "ForAll": " do", "While": " do", "Until": ""}[name]
                if name in ("ElsIf",):
                    ind -= 1
                if name == "Until":
                    ind -= 1
                emit(f"{word} {self.inline(c, True).strip()}{end}")
                if name != "Until":
                    ind += 1
                if tail.strip():
                    emit(self.inline(tail, True).strip())
            elif name == "Else":
                ind -= 1
                emit("else")
                ind += 1
                if rest.strip():
                    emit(self.inline(rest, True).strip())
            elif name in ("EndIf", "EndFor", "EndWhile", "EndLoop", "EndFunction", "EndProcedure"):
                ind -= 1
                emit({"EndIf": "end if", "EndFor": "end for", "EndWhile": "end while", "EndLoop": "end loop",
                      "EndFunction": "end function", "EndProcedure": "end procedure"}[name])
                if rest.strip():
                    emit(self.inline(rest, True).strip())
            elif name in ("Repeat", "Loop"):
                emit("repeat" if name == "Repeat" else "loop")
                ind += 1
                if rest.strip():
                    emit(self.inline(rest, True).strip())
            elif name in ("Function", "Procedure"):
                (f, a), tail = arg(2)
                emit(f"{name.lower()} {self.inline(f, True).strip()}({self.inline(a, True).strip()})")
                ind += 1
                if tail.strip():
                    emit(self.inline(tail, True).strip())
            elif name in ("Require", "Ensure"):
                emit(("Input: " if name == "Require" else "Output: ") + self.inline(rest, True).strip())
            elif name == "Return":
                emit("return " + self.inline(rest, True).strip())
        return [line.rstrip() for line in lines if line.strip()]

    def bibliography(self, body):
        _, _, e = read_args(body, 0, 0, 1)
        body = body[e:]
        out = ["## References", ""]
        parts = re.split(r"\\bibitem\{([^}]*)\}", body)
        for k in range(1, len(parts), 2):
            key, text = parts[k], parts[k + 1]
            out.append(f"{self.cites[key]}. {self.inline(text).strip()}")
        return out + [""]

    def convert(self):
        lines = self.blocks(self.body)
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
        return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("aux")
    ap.add_argument("out")
    ap.add_argument("--figdir", default="manual/fig")
    a = ap.parse_args()
    labels = read_aux(a.aux)
    src = open(a.source).read()
    # the guides' generated tables (make_tables.py), inlined as LaTeX inlines them
    src = re.sub(r"\\input\{(tables/[^}]+)\}",
                 lambda m: Path(Path(a.source).parent / m.group(1)).with_suffix(".tex").read_text(), src)
    conv = Converter(src, labels, a.figdir)
    md = conv.convert()
    open(a.out, "w").write(md)
    for w in sorted(set(WARN)):
        print("tex2md:", w, file=sys.stderr)
    print(f"tex2md: {a.out}: {len(md.splitlines())} lines, {len(set(WARN))} warnings", file=sys.stderr)


if __name__ == "__main__":
    main()
