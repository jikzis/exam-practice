# Exam Practice – Data Analytics in Accounting

A practice website for the oral exam in *Data Analytics in Accounting*. Students get the same two exam types as the real exam:

* **Data manipulation exams** (300 exams): 3 Excel + 3 SQL questions on two course files, with an 8-minute timer.
* **Data interpretation exams** (30 exams): regression output estimated on the restaurant case (Larry’s Commissary), with four questions each (prediction, significance, coefficient interpretation, R²) and a 5-minute timer.
* **Question bank** (220 questions), filterable by subject, file, skill and status.

All questions are built only from the course files in `../materials` (NYC film permits, ETL cases 1–4, IntegrateCo payroll, the restaurant regression data, the CS50 movies/songs databases, Chinook and the OSU P-card database).

## Added functionality: answer checking with mistake diagnosis

On the original website a data manipulation question has no answer key. The only help is an AI tutor, and the tutor receives the student’s *description* of what they did. It never executes anything on the 94 MB P-card database or the 33,502-row permit file, so neither the student nor the tutor can tell whether the result is right. My site adds a **“Check your result”** box to every question. The student types one or two checkpoint values (a count, a total, a name) and immediately sees whether they are correct. The answer key was produced by *executing* the model solution on the same file the student downloads: the SQL shown under “Show model solution” is the query that produced the key. That is the course’s own lesson that executed code, not a fluent explanation, should verify a calculation. The more important part is the **diagnosis**. Telling an accountant “wrong” in an 8-minute exam is not enough; they need to know *which* mistake they made. The generator therefore also executes the typical *wrong* approach for 286 known pitfalls and stores the value each one produces. Most of these pitfalls are named in the course materials themselves: refunds stored as negative amounts (2014 net spending of $33,504,148.34 vs. purchases of $34,133,593.60, a difference nobody spots by eye), FullName rather than last name as the cardholder identifier, `>=` vs `>` at the $5,000 P-card limit, PreparerID that only becomes unique together with BusinessUnitID, `RIGHT(Job,3)` failing on job codes with a location suffix, PlantID stored as text (in Excel text compares as larger than every number), Excel date-times stored as days rather than hours, case-sensitive `=` instead of `LIKE` for “Amazon”, the second Kevin Bacon, and a missing interaction term in a regression prediction. A student who submits one of those numbers is told *what* they did wrong, not only *that* they were wrong. A “My mistakes” page groups the recognised mistakes so students can see which trap they keep falling into. This is not on the AI’s list of suggestions. Items 13–24 polish a tutor that still cannot see the data, and item 66 validates content for the maintainer rather than giving feedback to the student. Research on feedback supports the design: feedback that explains the error helps learning more than a right/wrong verdict (Hattie & Timperley, 2007, *The Power of Feedback*; Shute, 2008, *Focus on Formative Feedback*). The checker also needs no AI calls, so it has no running cost and works offline, which makes the rate-limit and budget items (1–2) unnecessary for this feature.

## Why the starting materials make this case harder than the previous cases

In the earlier cases the input was one clean file and a clear task. Here the starting material is a heterogeneous course folder. It holds Word instructions, `.xlsx` workbooks, a macro-enabled `.xlsm`, “`.csv`” files that are really pipe-delimited, a text file with quoted commas, and SQLite databases of 69 MB and 94 MB. Nowhere does it contain a question bank or, apart from the PwC sheets, any answer key. Everything a practice site needs (questions, accepted answers, time estimates, solutions) has to be designed and then *derived* from the data. The data is also messy on purpose. “Apple, Inc.” has no text qualifier, values carry “M”/“K” suffixes and parentheses for negatives, codes mix “M” and “Mo”, and IDs are only unique in combination. Every generated answer therefore has to be verified by running code rather than trusted from an LLM’s reading of the files. The files are also far too large to paste into an LLM’s context (489,178 P-card rows, 1.2 million movie-star links), so the LLM has to write programs instead of reading the data. Finally, a working reference site already exists, so building a site is not the hard part. Deciding what is worth adding requires domain knowledge of the exam and of accounting data, which is the skill this case is meant to test.

## Project structure

```
exam-practice/
├── build/generate.py   # builds the question bank: runs every SQL solution, computes Excel answers with pandas,
│                       # executes the known mistakes, estimates the regressions (numpy/scipy)
├── site/               # static website (no server, no build step)
│   ├── index.html, styles.css, app.js
│   ├── bank.js         # generated question bank
│   └── datasets/       # downloadable copies of the course files
└── README.md
```

## Rebuild and run locally

```bash
python build/generate.py
```

```bash
python -m http.server 8765 --directory site
```

Then open http://localhost:8765. The generator needs Python 3.10+ with pandas, numpy, scipy and openpyxl.

**Self-test:** the generator asserts that every field has an answer. In the browser console, `window.__examPractice.checkField` can be run over `BANK` to confirm that every answer key is accepted and every stored mistake value triggers its diagnosis (366 answer fields and 529 diagnostics pass).
