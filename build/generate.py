"""
Builds the question bank for the Exam Practice site.

Every answer key is produced by executing code against the real course files:
  * SQL questions: the model-solution query itself is executed with sqlite3 and its
    result becomes the answer key, so the solution shown to students is guaranteed
    to produce the accepted answer.
  * Excel questions: the expected value is computed with pandas from the same file
    the student downloads.
  * Mistake diagnostics: each known pitfall (most of them named in the course
    materials themselves) is executed as a "wrong" query / calculation, and the
    value it produces is stored with an explanation. When a student submits that
    value, the site explains which mistake produces it.

Run:  python build/generate.py   (from the exam-practice folder)
Output: site/bank.js and site/datasets/*
"""

import json
import math
import random
import re
import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
MAT = ROOT.parent / "materials"
SITE = ROOT / "site"
DS = SITE / "datasets"

P = {
    "film": MAT / "1. Excel Part 1/PWC Excel Assignment/NYC_Film_Permits.txt",
    "etl1": MAT / "1. Excel part 2/ETL Assignment/data/case 1/Analytics_mindset_case_studies_ETL_Case1.csv",
    "etl2": MAT / "1. Excel part 2/ETL Assignment/data/case 2/Analytics_mindset_case_studies_ETL_Case2_Excel.xlsx",
    "etl3": MAT / "1. Excel part 2/ETL Assignment/data/case 3/Analytics_mindset_case_studies_ETL_Case3_Excel.xlsx",
    "etl4": MAT / "1. Excel part 2/ETL Assignment/data/case 4/Analytics_mindset_case_studies_ETL_Case4_Excel.xlsx",
    "pay15": MAT / "1. Excel part 3/IntegrateCo Assignment/Analytics_mindset_case_studies_IntegrateCo_2015payroll.csv",
    "pay16": MAT / "1. Excel part 3/IntegrateCo Assignment/Analytics_mindset_case_studies_IntegrateCo_2016payroll.csv",
    "jobs": MAT / "1. Excel part 3/IntegrateCo Assignment/Analytics_mindset_case_studies_IntegrateCo_jobcodes.csv",
    "locs": MAT / "1. Excel part 3/IntegrateCo Assignment/Analytics_mindset_case_studies_IntegrateCo_locations.csv",
    "rest": MAT / "1. Excel part 4/Regression Assignment/Restaurant/data.xlsm",
    "movies": MAT / "2. SQL part 1/Assignments/cs50 assignment/movies/movies.db",
    "songs": MAT / "2. SQL part 1/Assignments/cs50 assignment/songs/songs.db",
    "chinook": MAT / "2. SQL part 1/Assignments/pwc assignment/Chinook_Sqlite.sqlite",
    "pcards": MAT / "2. SQL part 2/Pcards Assignment/pcards.db",
}

FILES = {
    "film": dict(subject="Excel", label="NYC film permits", downloads=["film"],
                 note="Comma-delimited text file with a header row. Several text fields contain commas inside quotes.",
                 schema="EventID, EventType, StartDateTime, EndDateTime, EnteredOn, EventAgency, ParkingHeld, Borough, CommunityBoard(s), PolicePrecinct(s), Category, SubCategoryName, Country, ZipCode"),
    "etl1": dict(subject="Excel", label="ETL case 1 – balance sheets", downloads=["etl1"],
                 note="Messy balance-sheet extract. Do not verify the amounts themselves; fix the data problems.",
                 schema="Ticker, Name, Year, Cash, Accounts Receivables, Inventories, Total Current Assets, Total Depreciation"),
    "etl2": dict(subject="Excel", label="ETL case 2 – employee codes", downloads=["etl2"],
                 note="EmployeeCode = Location (3 letters) + EmpID + '-' + PlantID + PayPeriod (W or M).",
                 schema="Sheet 'Case 2': EmployeeCode, FirstName, LastName"),
    "etl3": dict(subject="Excel", label="ETL case 3 – messy employee codes", downloads=["etl3"],
                 note="Location codes are ARG, AUS, Canada, ENG, GER, Japan, MEX, US. PayPeriod is W/We (weekly) or M/Mo (monthly).",
                 schema="Sheet 'Case 3 data': EmployeeCode, FirstName, LastName"),
    "etl4": dict(subject="Excel", label="ETL case 4 – journal entries", downloads=["etl4"],
                 note="PreparerID restarts in every business unit: PreparerID + BusinessUnitID identifies a preparer.",
                 schema="Sheets: JELineItems, BusinessUnits, ChartOfAccounts, PreparerInfo, Source"),
    "pay": dict(subject="Excel", label="IntegrateCo payroll 2015/2016", downloads=["pay15", "pay16", "jobs", "locs"],
                note="Pipe-delimited (|) files saved as .csv. Job/Project = 2-digit start year + optional H/S + 3-digit job code + optional location code (B, R, SG, SH). Job codes are text: 001, not 1.",
                schema="2015: Pay_Date|Payroll_ID|Emp_No|Home_Div|Work_Div|First_Name|Last_Name|Job|Reg_Hrs|OT_Hrs|Reg_Pay|OT_Pay|Futa|Suta|Fica_Medc|Work_Comp · 2016: Project|Employee_ID|Employee_Number|Employee_Name|Hours|OT_Hours|Gross_Wages|SocSec_Medicare|FUTA|SUTA|Work_Comp"),
    "pcards": dict(subject="SQL", label="OSU P-cards", downloads=["pcards"],
                   note="Use the Year/Month fields for dates. Purchases have Amount > 0; refunds have Amount < 0. FullName identifies a cardholder.",
                   schema="pcards(Year, Month, FullName, ID, AgencyNumber, AgencyName, CardholderLastName, CardholderFirstInitial, Description, Amount, Vendor, TransactionDate, PostedDate, MCC)"),
    "movies": dict(subject="SQL", label="Movies (CS50)", downloads=["movies"],
                   note="Large IMDb extract. A person can appear many times in stars; names are not unique.",
                   schema="movies(id, title, year) · ratings(movie_id, rating, votes) · stars(movie_id, person_id) · directors(movie_id, person_id) · people(id, name, birth)"),
    "songs": dict(subject="SQL", label="Songs (CS50)", downloads=["songs"],
                  note="Small Spotify extract of 100 songs.",
                  schema="songs(id, name, artist_id, danceability, energy, key, loudness, speechiness, valence, tempo, duration_ms) · artists(id, name)"),
    "chinook": dict(subject="SQL", label="Chinook music store (PwC)", downloads=["chinook"],
                    note="Digital music store: customers buy tracks on invoices.",
                    schema="Customer, Employee, Invoice, InvoiceLine, Track, Album, Artist, Genre, MediaType, Playlist, PlaylistTrack"),
    "rest": dict(subject="Regression", label="Larry's Commissary (restaurant)", downloads=["rest"],
                 note="Daily point-of-sale data per menu item for 2018, plus weather and football game days.",
                 schema="Sheets: salesData, weatherData, gameData (+ data dictionaries)"),
}

QUESTIONS = []
_counter = {}


def num(x, d=2):
    return float(round(float(x), d))


def add(file, topics, prompt, seconds, fields, solution, why=None):
    """Register a question. fields: list of dicts with label/kind/answer/(tol)/(options)/(diag)."""
    subject = FILES[file]["subject"]
    pre = {"Excel": "X", "SQL": "S"}[subject]
    _counter[pre] = _counter.get(pre, 0) + 1
    qid = f"{pre}{_counter[pre]:03d}"
    clean = []
    for f in fields:
        f = dict(f)
        kind = f.setdefault("kind", "number")
        if kind in ("number", "percent"):
            f["answer"] = float(f["answer"])
            f.setdefault("tol", default_tol(f["answer"], kind))
        diags = []
        seen = set()
        for value, msg in f.pop("diag", []):
            if value is None:
                continue
            if kind in ("number", "percent"):
                if isinstance(value, float) and math.isnan(value):
                    continue
                value = float(value)
                if abs(value - f["answer"]) <= f["tol"]:
                    continue  # the mistake gives the same result here, so it is not diagnosable
                key = round(value, 6)
            else:
                value = str(value)
                if norm_text(value) == norm_text(f["answer"]):
                    continue
                key = norm_text(value)
            if key in seen:
                continue
            seen.add(key)
            diags.append({"value": value, "msg": msg})
        f["diagnostics"] = diags
        clean.append(f)
    q = dict(id=qid, subject=subject, file=file, topics=topics, prompt=prompt, seconds=seconds,
             fields=clean, solution=solution)
    if why:
        q["why"] = why
    QUESTIONS.append(q)
    return q


def default_tol(a, kind):
    if kind == "percent":
        return 0.0006  # answers stored as fractions; 0.06 percentage points
    a = abs(a)
    if a == int(a) and a < 1e6:
        return 0.0
    # two-decimal answers: accept rounding to the cent, and for large totals rounding to the whole unit
    return max(0.006, min(0.51, a * 1e-6))


def norm_text(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


# ----------------------------------------------------------------------------- SQL helpers
def sql_runner(path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

    def one(sql):
        r = con.execute(sql).fetchone()
        return None if r is None else (r[0] if len(r) == 1 else r)

    def rows(sql):
        return con.execute(sql).fetchall()

    return con, one, rows


def sqlsol(sql, note=None):
    return {"type": "sql", "code": " ".join(sql.split()) if "\n" not in sql else sql.strip(), "note": note}


# ============================================================================= SQL: P-cards
def build_pcards():
    con, one, rows = sql_runner(P["pcards"])
    for y in [2011, 2012, 2013, 2014]:
        sql = f"SELECT COUNT(*) FROM pcards WHERE Year = {y} AND Amount > 0;"
        ans = one(sql)
        add("pcards", ["filtering", "aggregation"],
            f"How many purchases (Amount > 0) were made in {y}?", 60,
            [dict(label="Number of purchases", answer=ans, diag=[
                (one(f"SELECT COUNT(*) FROM pcards WHERE Year = {y}"), "This is every transaction in the year. Refunds (Amount < 0) are not purchases and must be filtered out."),
                (one("SELECT COUNT(*) FROM pcards WHERE Amount > 0"), "This count covers all years. Add the condition on Year."),
            ])], sqlsol(sql))

        sql = f"SELECT ROUND(SUM(Amount), 2) FROM pcards WHERE Year = {y} AND Amount > 0;"
        add("pcards", ["filtering", "aggregation"],
            f"What is the total value of purchases (Amount > 0) in {y}? Round to two decimals.", 75,
            [dict(label="Total purchases ($)", answer=one(sql), diag=[
                (one(f"SELECT ROUND(SUM(Amount),2) FROM pcards WHERE Year = {y}"), "This is net spending: refunds were subtracted. The question asks for purchases only (Amount > 0)."),
                (one(f"SELECT ROUND(SUM(ABS(Amount)),2) FROM pcards WHERE Year = {y}"), "Refunds were added as if they were purchases (absolute values). Filter on Amount > 0 instead."),
            ])], sqlsol(sql))

        sql = f"SELECT COUNT(*) FROM pcards WHERE Year = {y} AND Amount > 5000;"
        add("pcards", ["filtering", "internal controls"],
            f"The single-transaction limit is $5,000. How many transactions in {y} were for more than $5,000?", 60,
            [dict(label="Transactions over $5,000", answer=one(sql), diag=[
                (one(f"SELECT COUNT(*) FROM pcards WHERE Year = {y} AND Amount >= 5000"), "You used >= 5000. A transaction of exactly $5,000 is within the limit, so use > 5000."),
                (one(f"SELECT COUNT(*) FROM pcards WHERE Year = {y} AND ABS(Amount) > 5000"), "Refunds over $5,000 are included. A refund does not break the purchase limit."),
            ])], sqlsol(sql))

        sql = (f"SELECT FullName, COUNT(*) AS NumTrans FROM pcards WHERE Year = {y} "
               f"GROUP BY FullName ORDER BY NumTrans DESC LIMIT 1;")
        name, cnt = one(sql)
        wrong_ln = one(f"SELECT CardholderLastName, COUNT(*) c FROM pcards WHERE Year={y} GROUP BY CardholderLastName ORDER BY c DESC LIMIT 1")
        add("pcards", ["grouping", "sorting"],
            f"Which cardholder made the most transactions in {y}, and how many?", 90,
            [dict(label="Cardholder (FullName)", kind="text", answer=name,
                  diag=[(wrong_ln[0], "You grouped by last name only. Different people share last names, so group by FullName (first initial + last name).")]),
             dict(label="Number of transactions", answer=cnt,
                  diag=[(wrong_ln[1], "This count belongs to the last name, which combines several people. Group by FullName.")])],
            sqlsol(sql))

        sql = f"SELECT ROUND(SUM(Amount), 2) FROM pcards WHERE Year = {y} AND Amount < 0;"
        add("pcards", ["filtering", "aggregation"],
            f"What is the total value of refunds in {y}? Report it as a negative number, rounded to two decimals.", 60,
            [dict(label="Total refunds ($)", answer=one(sql), diag=[
                (-one(sql), "Refunds are stored as negative amounts. Keep the minus sign."),
                (one(f"SELECT ROUND(SUM(Amount),2) FROM pcards WHERE Year={y} AND Amount <= 0"), "Amount <= 0 gives the same sum, so this message should not appear."),
            ])], sqlsol(sql))

        sql = f"SELECT ROUND(SUM(Amount) * 0.045, 2) FROM pcards WHERE Year = {y} AND Amount > 0;"
        add("pcards", ["calculated fields"],
            f"OSU is exempt from the 4.5% state sales tax. How much sales tax did it avoid on purchases (Amount > 0) in {y}? Round to two decimals.", 90,
            [dict(label="Sales tax saved ($)", answer=one(sql), diag=[
                (one(f"SELECT ROUND(SUM(Amount)*0.045,2) FROM pcards WHERE Year={y}"), "Refunds were included. Only purchases (Amount > 0) would have carried sales tax."),
                (one(f"SELECT ROUND(SUM(Amount)*4.5,2) FROM pcards WHERE Year={y} AND Amount>0"), "You multiplied by 4.5 instead of 0.045 (4.5%)."),
                (one(f"SELECT ROUND(SUM(Amount)*1.045,2) FROM pcards WHERE Year={y} AND Amount>0"), "This is the total including tax. The question asks for the tax amount only (× 0.045)."),
            ])], sqlsol(sql))

        sql = (f"SELECT COUNT(*) FROM (SELECT FullName FROM pcards WHERE Year = {y} "
               f"GROUP BY FullName HAVING SUM(Amount) > 50000);")
        add("pcards", ["grouping", "HAVING", "internal controls"],
            f"Cardholders must not spend more than $50,000 per year. How many cardholders spent more than $50,000 (net of refunds) in {y}?", 90,
            [dict(label="Number of cardholders", answer=one(sql), diag=[
                (one(f"SELECT COUNT(*) FROM pcards WHERE Year={y} AND Amount > 50000"), "You counted single transactions above $50,000. The limit is per cardholder per year: GROUP BY FullName and HAVING SUM(Amount) > 50000."),
                (one(f"SELECT COUNT(*) FROM (SELECT CardholderLastName FROM pcards WHERE Year={y} GROUP BY CardholderLastName HAVING SUM(Amount)>50000)"), "You grouped by last name, which merges different people. Group by FullName."),
            ])], sqlsol(sql))

        sql = (f"SELECT MCC, ROUND(SUM(Amount), 2) AS Total FROM pcards WHERE Year = {y} "
               f"GROUP BY MCC ORDER BY Total DESC LIMIT 1;")
        mcc, tot = one(sql)
        wrong = one(f"SELECT MCC, COUNT(*) c FROM pcards WHERE Year={y} GROUP BY MCC ORDER BY c DESC LIMIT 1")
        add("pcards", ["grouping", "sorting"],
            f"Which merchant category (MCC) had the highest net spending in {y}, and what was the total?", 90,
            [dict(label="MCC", kind="text", answer=mcc,
                  diag=[(wrong[0], "This is the MCC with the most transactions, not the highest spending. Order by SUM(Amount).")]),
             dict(label="Total ($)", answer=tot)], sqlsol(sql))

    for y, m in [(2014, 1), (2014, 6), (2013, 9), (2012, 11), (2011, 3)]:
        sql = (f"SELECT COUNT(*) FROM (SELECT FullName FROM pcards WHERE Year = {y} AND Month = {m} "
               f"GROUP BY FullName HAVING SUM(Amount) > 10000);")
        add("pcards", ["grouping", "HAVING", "internal controls"],
            f"Cardholders need approval to spend more than $10,000 per month. How many cardholders spent more than $10,000 (net) in month {m} of {y}?", 90,
            [dict(label="Number of cardholders", answer=one(sql), diag=[
                (one(f"SELECT COUNT(*) FROM pcards WHERE Year={y} AND Month={m} AND Amount > 10000"), "You counted single transactions. The limit is per cardholder per month: GROUP BY FullName with HAVING."),
                (one(f"SELECT COUNT(*) FROM (SELECT FullName FROM pcards WHERE Year={y} GROUP BY FullName HAVING SUM(Amount)>10000)"), "The month filter is missing, so this covers the whole year."),
            ])], sqlsol(sql))

        sql = f"SELECT ROUND(SUM(Amount), 2) FROM pcards WHERE Year = {y} AND Month = {m} AND Amount < 0;"
        add("pcards", ["filtering"],
            f"What was the total value of refunds in month {m} of {y}? Report it as a negative number.", 60,
            [dict(label="Refunds ($)", answer=one(sql), diag=[
                (-one(sql), "Keep the minus sign: refunds are stored as negative amounts."),
                (one(f"SELECT ROUND(SUM(Amount),2) FROM pcards WHERE Year={y} AND Amount<0"), "This is the refund total for the whole year. Add Month = " + str(m) + "."),
            ])], sqlsol(sql))

    for y in [2012, 2013, 2014]:
        sql = f"SELECT COUNT(*) FROM pcards WHERE Year = {y} AND Vendor LIKE '%amazon%';"
        add("pcards", ["text search", "LIKE"],
            f"How many transactions in {y} were with a vendor that has “Amazon” anywhere in its name (any capitalisation)?", 75,
            [dict(label="Transactions", answer=one(sql), diag=[
                (one(f"SELECT COUNT(*) FROM pcards WHERE Year={y} AND Vendor = 'Amazon.com'"), "You matched the exact text 'Amazon.com'. Vendor names vary (AMAZON MKTPLACE PMTS, Amazon Services-Kindle …). Use LIKE '%amazon%'."),
                (one(f"SELECT COUNT(*) FROM pcards WHERE Year={y} AND Vendor LIKE 'amazon%'"), "'amazon%' only finds names that start with Amazon. Put % on both sides."),
                (one(f"SELECT COUNT(*) FROM pcards WHERE Year={y} AND Vendor GLOB '*Amazon*'"), "GLOB and = are case-sensitive in SQLite, so AMAZON in capitals was missed. LIKE ignores case."),
            ])], sqlsol(sql))

    # split purchases & duplicates & Benford (2014 / 2013)
    for y in [2013, 2014]:
        sql = (f"SELECT COUNT(*) FROM (SELECT FullName, Vendor, TransactionDate FROM pcards\n"
               f"  WHERE Year = {y} AND Amount > 0\n"
               f"  GROUP BY FullName, Vendor, TransactionDate\n"
               f"  HAVING COUNT(*) > 1 AND SUM(Amount) > 5000);")
        add("pcards", ["grouping", "HAVING", "internal controls"],
            f"Split purchases: how many combinations of cardholder, vendor and transaction date in {y} have more than one purchase that together exceed $5,000?", 150,
            [dict(label="Number of combinations", answer=one(sql), diag=[
                (one(f"SELECT COUNT(*) FROM (SELECT 1 FROM pcards WHERE Year={y} AND Amount>0 GROUP BY FullName,Vendor,TransactionDate HAVING SUM(Amount)>5000)"), "COUNT(*) > 1 is missing. A single large purchase is not a split purchase."),
                (one(f"SELECT COUNT(*) FROM (SELECT 1 FROM pcards WHERE Year={y} GROUP BY FullName,Vendor,TransactionDate HAVING COUNT(*)>1 AND SUM(Amount)>5000)"), "Refunds are included in the groups. Filter on Amount > 0 first."),
                (one(f"SELECT COUNT(*) FROM (SELECT 1 FROM pcards WHERE Year={y} AND Amount>0 GROUP BY FullName,TransactionDate HAVING COUNT(*)>1 AND SUM(Amount)>5000)"), "Vendor is not in the GROUP BY, so purchases from different vendors were combined."),
            ])], sqlsol(sql), why="Why must the GROUP BY include the vendor, the cardholder and the date?")

        sql = (f"SELECT COUNT(*) FROM (SELECT TransactionDate, Vendor, FullName, Amount FROM pcards\n"
               f"  WHERE Year = {y} AND Amount > 0\n"
               f"  GROUP BY TransactionDate, Vendor, FullName, Amount\n"
               f"  HAVING COUNT(*) > 1);")
        add("pcards", ["grouping", "HAVING", "forensics"],
            f"Possible duplicate payments: how many groups in {y} have the same date, vendor, cardholder and amount appearing more than once (purchases only)?", 120,
            [dict(label="Number of duplicate groups", answer=one(sql), diag=[
                (one(f"SELECT SUM(c) FROM (SELECT COUNT(*) c FROM pcards WHERE Year={y} AND Amount>0 GROUP BY TransactionDate,Vendor,FullName,Amount HAVING COUNT(*)>1)"), "You counted the duplicated transactions, not the groups. Count the rows returned by the grouped query."),
                (one(f"SELECT COUNT(*) FROM (SELECT 1 FROM pcards WHERE Year={y} GROUP BY TransactionDate,Vendor,FullName,Amount HAVING COUNT(*)>1)"), "Refunds are included. The question covers purchases only."),
            ])], sqlsol(sql))

        sql = (f"SELECT ROUND(100.0 * SUM(SUBSTR(CAST(Amount AS TEXT), 1, 1) = '1') / COUNT(*), 3)\n"
               f"FROM pcards WHERE Year = {y} AND Amount >= 1;")
        add("pcards", ["calculated fields", "forensics"],
            f"Benford’s law: what percentage of {y} transactions with Amount ≥ 1 start with the digit 1? Round to three decimals.", 150,
            [dict(label="Percentage starting with 1 (%)", answer=one(sql), tol=0.0006, diag=[
                (one(f"SELECT ROUND(100.0*SUM(SUBSTR(CAST(ABS(Amount) AS TEXT),1,1)='1')/COUNT(*),3) FROM pcards WHERE Year={y} AND ABS(Amount)>=1"), "Refunds were included (absolute values). Use Amount >= 1 only."),
                (one(f"SELECT ROUND(100.0*SUM(SUBSTR(CAST(Amount AS TEXT),1,1)='1')/COUNT(*),3) FROM pcards WHERE Year={y}"), "The denominator includes amounts below 1 and refunds. Filter Amount >= 1 before counting."),
            ])], sqlsol(sql, "Benford expects about 30.1% for the digit 1."))
    con.close()


# ============================================================================= SQL: movies
def build_movies():
    con, one, rows = sql_runner(P["movies"])
    for y in [2005, 2010, 2013, 2016, 2018]:
        sql = f"SELECT COUNT(*) FROM movies WHERE year = {y};"
        add("movies", ["filtering"], f"How many movies were released in {y}?", 45,
            [dict(label="Movies", answer=one(sql), diag=[
                (one(f"SELECT COUNT(*) FROM movies WHERE year >= {y}"), f"You used >= {y}, which includes every later year."),
                (one(f"SELECT COUNT(*) FROM movies m JOIN ratings r ON r.movie_id=m.id WHERE year={y}"), "The join to ratings drops movies without a rating. The question asks about all movies."),
            ])], sqlsol(sql))

        sql = f"SELECT ROUND(AVG(r.rating), 2) FROM movies m JOIN ratings r ON r.movie_id = m.id WHERE m.year = {y};"
        add("movies", ["joins", "aggregation"], f"What is the average rating of movies released in {y}? Round to two decimals.", 75,
            [dict(label="Average rating", answer=one(sql), diag=[
                (one("SELECT ROUND(AVG(rating),2) FROM ratings"), "This is the average over all years. Join movies and filter on year."),
                (one(f"SELECT ROUND(SUM(r.rating*r.votes)*1.0/SUM(r.votes),2) FROM movies m JOIN ratings r ON r.movie_id=m.id WHERE m.year={y}"), "You weighted by votes. The question asks for the plain average of the ratings."),
            ])], sqlsol(sql))

        sql = f"SELECT COUNT(*) FROM movies m JOIN ratings r ON r.movie_id = m.id WHERE m.year = {y} AND r.rating >= 8.0;"
        add("movies", ["joins", "filtering"], f"How many movies released in {y} have a rating of at least 8.0?", 60,
            [dict(label="Movies", answer=one(sql), diag=[
                (one(f"SELECT COUNT(*) FROM movies m JOIN ratings r ON r.movie_id=m.id WHERE m.year={y} AND r.rating>8"), "You used > 8.0. “At least” means >= 8.0."),
            ])], sqlsol(sql))

        sql = (f"SELECT COUNT(DISTINCT s.person_id) FROM stars s JOIN movies m ON m.id = s.movie_id WHERE m.year = {y};")
        add("movies", ["joins", "DISTINCT"], f"How many different people starred in movies released in {y}?", 75,
            [dict(label="People", answer=one(sql), diag=[
                (one(f"SELECT COUNT(*) FROM stars s JOIN movies m ON m.id=s.movie_id WHERE m.year={y}"), "Without DISTINCT, people who starred in several movies are counted several times."),
                (one(f"SELECT COUNT(DISTINCT p.name) FROM stars s JOIN movies m ON m.id=s.movie_id JOIN people p ON p.id=s.person_id WHERE m.year={y}"), "You counted distinct names. Names are not unique: count distinct person_id."),
            ])], sqlsol(sql))

    for name in ["Johnny Depp", "Helena Bonham Carter", "Tom Hanks", "Meryl Streep", "Morgan Freeman", "Scarlett Johansson"]:
        sql = (f"SELECT COUNT(*) FROM stars s JOIN people p ON p.id = s.person_id WHERE p.name = '{name}';")
        ids = rows(f"SELECT id FROM people WHERE name='{name}'")
        if len(ids) != 1:
            continue
        add("movies", ["joins"], f"In how many movies in the database did {name} star? (Assume there is only one {name}.)", 60,
            [dict(label="Movies", answer=one(sql), diag=[
                (one(f"SELECT COUNT(*) FROM directors d JOIN people p ON p.id=d.person_id WHERE p.name='{name}'"), "You joined directors instead of stars."),
            ])], sqlsol(sql))

    sql = ("SELECT COUNT(DISTINCT s.person_id) FROM stars s\n"
           "WHERE s.movie_id IN (SELECT movie_id FROM stars WHERE person_id =\n"
           "        (SELECT id FROM people WHERE name = 'Kevin Bacon' AND birth = 1958))\n"
           "  AND s.person_id != (SELECT id FROM people WHERE name = 'Kevin Bacon' AND birth = 1958);")
    add("movies", ["subqueries", "DISTINCT"],
        "How many different people starred in a movie together with Kevin Bacon (born 1958)? Do not count Kevin Bacon himself.", 180,
        [dict(label="People", answer=one(sql), diag=[
            (one("SELECT COUNT(DISTINCT person_id) FROM stars WHERE movie_id IN (SELECT movie_id FROM stars WHERE person_id=(SELECT id FROM people WHERE name='Kevin Bacon' AND birth=1958))"), "Kevin Bacon himself is included. Exclude his person_id."),
            (one("SELECT COUNT(DISTINCT person_id) FROM stars WHERE movie_id IN (SELECT movie_id FROM stars WHERE person_id IN (SELECT id FROM people WHERE name='Kevin Bacon')) AND person_id NOT IN (SELECT id FROM people WHERE name='Kevin Bacon')"), "There are two people called Kevin Bacon. Filter on birth = 1958."),
            (one("SELECT COUNT(*) FROM stars WHERE movie_id IN (SELECT movie_id FROM stars WHERE person_id=(SELECT id FROM people WHERE name='Kevin Bacon' AND birth=1958)) AND person_id != (SELECT id FROM people WHERE name='Kevin Bacon' AND birth=1958)"), "DISTINCT is missing, so co-stars from several movies are counted repeatedly."),
        ])], sqlsol(sql), why="Why is filtering on birth = 1958 necessary here?")

    sql = ("SELECT COUNT(*) FROM movies WHERE id IN\n"
           "  (SELECT movie_id FROM stars WHERE person_id = (SELECT id FROM people WHERE name = 'Johnny Depp'))\n"
           "AND id IN\n"
           "  (SELECT movie_id FROM stars WHERE person_id = (SELECT id FROM people WHERE name = 'Helena Bonham Carter'));")
    add("movies", ["subqueries"], "In how many movies did both Johnny Depp and Helena Bonham Carter star?", 150,
        [dict(label="Movies", answer=one(sql), diag=[
            (one("SELECT COUNT(DISTINCT movie_id) FROM stars WHERE person_id IN (SELECT id FROM people WHERE name IN ('Johnny Depp','Helena Bonham Carter'))"), "This counts movies with either actor (OR). Both must star (intersect the two sets)."),
        ])], sqlsol(sql))

    for thr in [9.0, 9.5]:
        sql = (f"SELECT COUNT(DISTINCT d.person_id) FROM directors d JOIN ratings r ON r.movie_id = d.movie_id WHERE r.rating >= {thr};")
        add("movies", ["joins", "DISTINCT"], f"How many different people have directed a movie with a rating of at least {thr}?", 90,
            [dict(label="Directors", answer=one(sql), diag=[
                (one(f"SELECT COUNT(*) FROM directors d JOIN ratings r ON r.movie_id=d.movie_id WHERE r.rating>={thr}"), "Without DISTINCT, a director with several such movies is counted several times."),
                (one(f"SELECT COUNT(DISTINCT d.person_id) FROM directors d JOIN ratings r ON r.movie_id=d.movie_id WHERE r.rating>{thr}"), f"You used > {thr}. “At least” means >=."),
            ])], sqlsol(sql))

    for t in ["Toy Story", "Harry Potter", "Star Wars"]:
        sql = f"SELECT COUNT(*), MIN(year) FROM movies WHERE title LIKE '{t}%';"
        c, mn = one(sql)
        add("movies", ["LIKE"], f"How many movies have a title starting with “{t}”, and in which year was the first one released?", 60,
            [dict(label="Movies", answer=c, diag=[
                (one(f"SELECT COUNT(*) FROM movies WHERE title LIKE '%{t}%'"), "'%…%' also matches titles that contain the words later on. The question asks for titles starting with them."),
                (one(f"SELECT COUNT(*) FROM movies WHERE title = '{t}'"), "= only matches the exact title. Use LIKE with %."),
            ]), dict(label="First release year", answer=mn)], sqlsol(sql))
    con.close()


# ============================================================================= SQL: songs
def build_songs():
    con, one, rows = sql_runner(P["songs"])
    for a in ["Post Malone", "Drake", "XXXTENTACION", "Ed Sheeran", "Marshmello"]:
        sql = f"SELECT ROUND(AVG(energy), 3) FROM songs WHERE artist_id = (SELECT id FROM artists WHERE name = '{a}');"
        add("songs", ["subqueries", "aggregation"], f"What is the average energy of songs by {a}? Round to three decimals.", 75,
            [dict(label="Average energy", answer=one(sql), tol=0.0006, diag=[
                (one("SELECT ROUND(AVG(energy),3) FROM songs"), "This is the average for all songs. Filter on the artist."),
            ])], sqlsol(sql, "Do not hard-code the artist_id: look it up with a subquery or a join."))
        sql = f"SELECT COUNT(*) FROM songs s JOIN artists a ON a.id = s.artist_id WHERE a.name = '{a}';"
        add("songs", ["joins"], f"How many songs by {a} are in the database?", 45,
            [dict(label="Songs", answer=one(sql))], sqlsol(sql))
    sql = "SELECT COUNT(*) FROM songs WHERE name LIKE '%feat.%';"
    add("songs", ["LIKE"], "How many songs feature other artists (the name contains “feat.”)?", 45,
        [dict(label="Songs", answer=one(sql), diag=[(one("SELECT COUNT(*) FROM songs WHERE name LIKE 'feat.%'"), "'feat.%' only finds names starting with feat. Use '%feat.%'.")])], sqlsol(sql))
    for t in [0.6, 0.7, 0.75]:
        sql = f"SELECT COUNT(*) FROM songs WHERE danceability > {t} AND energy > {t} AND valence > {t};"
        add("songs", ["filtering"], f"How many songs have danceability, energy and valence all greater than {t}?", 60,
            [dict(label="Songs", answer=one(sql), diag=[
                (one(f"SELECT COUNT(*) FROM songs WHERE danceability > {t} OR energy > {t} OR valence > {t}"), "You used OR. All three conditions must hold: use AND."),
                (one(f"SELECT COUNT(*) FROM songs WHERE danceability >= {t} AND energy >= {t} AND valence >= {t}"), "You used >=. The question says greater than."),
            ])], sqlsol(sql))
    sql = "SELECT name, ROUND(duration_ms / 60000.0, 2) FROM songs ORDER BY duration_ms DESC LIMIT 1;"
    n, mins = one(sql)
    add("songs", ["sorting", "calculated fields"], "Which song is the longest, and how long is it in minutes (two decimals)?", 75,
        [dict(label="Song", kind="text", answer=n, diag=[(one("SELECT name FROM songs ORDER BY duration_ms ASC LIMIT 1"), "This is the shortest song: sort in DESC order.")]),
         dict(label="Minutes", answer=mins, diag=[
             (one("SELECT MAX(duration_ms) FROM songs"), "This is in milliseconds. Divide by 60,000."),
             (one("SELECT ROUND(MAX(duration_ms)/1000.0,2) FROM songs"), "This is in seconds. Divide milliseconds by 60,000 to get minutes."),
             (one("SELECT MAX(duration_ms)/60000 FROM songs"), "Integer division dropped the decimals. Divide by 60000.0."),
         ])], sqlsol(sql))
    sql = "SELECT ROUND(AVG(tempo), 2) FROM songs;"
    add("songs", ["aggregation"], "What is the average tempo of all songs? Round to two decimals.", 30,
        [dict(label="Average tempo", answer=one(sql))], sqlsol(sql))
    con.close()


# ============================================================================= SQL: Chinook
def build_chinook():
    con, one, rows = sql_runner(P["chinook"])
    for x in [10, 15, 20]:
        sql = f"SELECT COUNT(*) FROM Invoice WHERE Total > {x};"
        add("chinook", ["filtering"], f"How many invoices have a total greater than ${x}?", 45,
            [dict(label="Invoices", answer=one(sql), diag=[(one(f"SELECT COUNT(*) FROM Invoice WHERE Total >= {x}"), "You used >=. Greater than means >.")])], sqlsol(sql))
    for lo, hi in [(5, 10), (10, 20)]:
        sql = f"SELECT COUNT(*) FROM Invoice WHERE Total > {lo} AND Total < {hi};"
        add("chinook", ["filtering"], f"How many invoices have totals greater than ${lo} and less than ${hi}?", 60,
            [dict(label="Invoices", answer=one(sql), diag=[(one(f"SELECT COUNT(*) FROM Invoice WHERE Total BETWEEN {lo} AND {hi}"), "BETWEEN includes both end points. The question asks for strictly greater than and less than.")])], sqlsol(sql))
    for countries in [("Brazil", "Argentina", "Chile"), ("Germany", "France"), ("USA", "Canada")]:
        lst = ", ".join(f"'{c}'" for c in countries)
        sql = f"SELECT COUNT(*) FROM Invoice WHERE BillingCountry IN ({lst});"
        add("chinook", ["filtering", "IN"], f"How many invoices have a billing country of {', '.join(countries[:-1])} or {countries[-1]}?", 60,
            [dict(label="Invoices", answer=one(sql), diag=[(one(f"SELECT COUNT(*) FROM Customer WHERE Country IN ({lst})"), "You counted customers, not invoices.")])], sqlsol(sql))
    for letter in ["C", "B", "U"]:
        sql = f"SELECT COUNT(*) FROM Invoice WHERE BillingCountry LIKE '{letter}%';"
        add("chinook", ["LIKE"], f"How many invoices have a billing country that starts with “{letter}”?", 45,
            [dict(label="Invoices", answer=one(sql), diag=[(one(f"SELECT COUNT(*) FROM Invoice WHERE BillingCountry LIKE '%{letter}%'"), "'%…%' matches the letter anywhere in the name. Use '" + letter + "%'.")])], sqlsol(sql))
    sql = "SELECT COUNT(Company) FROM Customer;"
    add("chinook", ["aggregation", "NULL"], "How many customers list their company?", 45,
        [dict(label="Customers", answer=one(sql), diag=[(one("SELECT COUNT(*) FROM Customer"), "COUNT(*) counts every row. COUNT(Company) skips NULLs, or add WHERE Company IS NOT NULL.")])], sqlsol(sql))
    sql = "SELECT COUNT(*) FROM Employee WHERE strftime('%w', BirthDate) = '0';"
    add("chinook", ["dates"], "How many employees were born on a Sunday?", 90,
        [dict(label="Employees", answer=one(sql), diag=[(one("SELECT COUNT(*) FROM Employee WHERE strftime('%w',BirthDate)='7'"), "In SQLite strftime('%w') returns 0 for Sunday, 1 for Monday … 6 for Saturday.")])], sqlsol(sql))
    sql = "SELECT MAX(LENGTH(LastName)) FROM Customer;"
    add("chinook", ["text functions"], "How many characters long is the longest customer last name?", 45,
        [dict(label="Characters", answer=one(sql))], sqlsol(sql))
    sql = "SELECT BillingCountry, ROUND(SUM(Total), 2) AS Revenue FROM Invoice GROUP BY BillingCountry ORDER BY Revenue DESC LIMIT 1;"
    c, v = one(sql)
    add("chinook", ["grouping", "sorting"], "Which billing country has the highest total invoice value, and what is that total?", 75,
        [dict(label="Country", kind="text", answer=c, diag=[(one("SELECT BillingCountry FROM Invoice GROUP BY BillingCountry ORDER BY COUNT(*) DESC LIMIT 1"), "Same country, but check you ordered by SUM(Total), not by count.")]),
         dict(label="Total ($)", answer=v, diag=[(one("SELECT COUNT(*) FROM Invoice WHERE BillingCountry='USA'"), "This is the number of invoices. The question asks for SUM(Total).")])], sqlsol(sql))
    sql = ("SELECT g.Name, COUNT(*) AS LinesSold FROM InvoiceLine il JOIN Track t ON t.TrackId = il.TrackId\n"
           "JOIN Genre g ON g.GenreId = t.GenreId GROUP BY g.Name ORDER BY LinesSold DESC LIMIT 1;")
    g, n = one(sql)
    add("chinook", ["joins", "grouping"], "What is the most popular genre measured by invoice lines sold, and how many lines were sold?", 120,
        [dict(label="Genre", kind="text", answer=g),
         dict(label="Invoice lines", answer=n, diag=[(one("SELECT COUNT(*) FROM Track t JOIN Genre g ON g.GenreId=t.GenreId WHERE g.Name='Rock'"), "This counts tracks in the catalogue, not tracks sold. Start from InvoiceLine.")])], sqlsol(sql))
    sql = "SELECT COUNT(DISTINCT UnitPrice) FROM InvoiceLine;"
    add("chinook", ["DISTINCT"], "How many distinct unit prices appear on invoice lines?", 30,
        [dict(label="Distinct prices", answer=one(sql), diag=[(one("SELECT COUNT(UnitPrice) FROM InvoiceLine"), "DISTINCT is missing.")])], sqlsol(sql))
    for genre in ["Jazz", "Metal", "Latin", "Blues"]:
        sql = (f"SELECT ROUND(SUM(il.UnitPrice * il.Quantity), 2) FROM InvoiceLine il JOIN Track t ON t.TrackId = il.TrackId\n"
               f"JOIN Genre g ON g.GenreId = t.GenreId WHERE g.Name = '{genre}';")
        add("chinook", ["joins", "calculated fields"], f"What is the total sales value (UnitPrice × Quantity) of {genre} tracks?", 120,
            [dict(label="Sales ($)", answer=one(sql), diag=[
                (one(f"SELECT ROUND(SUM(t.UnitPrice),2) FROM Track t JOIN Genre g ON g.GenreId=t.GenreId WHERE g.Name='{genre}'"), "This is the list price of the whole catalogue for the genre. Start from InvoiceLine (what was sold)."),
            ])], sqlsol(sql))
    sql = "SELECT ROUND(MAX((julianday(HireDate) - julianday(BirthDate)) / 365.25), 1) FROM Employee;"
    add("chinook", ["dates", "calculated fields"], "How old (in years, one decimal) was the oldest employee when hired?", 90,
        [dict(label="Age at hire (years)", answer=one(sql), tol=0.06, diag=[
            (one("SELECT MAX(julianday(HireDate)-julianday(BirthDate)) FROM Employee"), "This is in days. Divide by 365.25."),
        ])], sqlsol(sql))
    con.close()


# ============================================================================= Excel: NYC film permits
def build_film():
    f = pd.read_csv(P["film"], dtype=str)
    for c in ["StartDateTime", "EndDateTime", "EnteredOn"]:
        f[c] = pd.to_datetime(f[c], format="%m/%d/%Y %H:%M")
    shoot = f[f.EventType == "Shooting Permit"].copy()
    shoot["dur_h"] = (shoot.EndDateTime - shoot.StartDateTime).dt.total_seconds() / 3600
    shoot["lead_d"] = (shoot.StartDateTime - shoot.EnteredOn).dt.total_seconds() / 86400

    add("film", ["import", "counting"], "Import the permit file. How many records (rows, excluding the header) and how many columns does it have?", 60,
        [dict(label="Records", answer=len(f), diag=[(len(f) + 1, "You counted the header row as a record.")]),
         dict(label="Columns", answer=f.shape[1], diag=[(f.shape[1] + 1, "A comma inside a quoted field was treated as a delimiter. Import with “ as the text qualifier.")])],
        {"type": "excel", "steps": ["Data ▸ From Text/CSV, delimiter comma, text qualifier \".", "Records: =ROWS(Table1) or =COUNTA(A:A)-1.", "Columns: =COLUMNS(Table1)."]})
    add("film", ["data quality", "COUNTBLANK"], "How many records have a missing ZipCode?", 45,
        [dict(label="Missing ZipCode", answer=int(f.ZipCode.isna().sum()))],
        {"type": "excel", "steps": ["=COUNTBLANK(N2:N33503) on the ZipCode column."]})
    vc = f.EventType.value_counts()
    add("film", ["pivot tables"], "How many different event types are there, and how many records does the most common one have?", 60,
        [dict(label="Event types", answer=len(vc)), dict(label="Records of most common type", answer=int(vc.iloc[0]), diag=[(len(f), "This is the total number of records.")])],
        {"type": "excel", "steps": ["PivotTable: EventType in Rows, Count of EventID in Values, sort descending."]})

    for b in ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]:
        sb = shoot[shoot.Borough == b]
        add("film", ["COUNTIFS"], f"How many shooting permits (EventType = Shooting Permit) are there in {b}?", 45,
            [dict(label="Shooting permits", answer=len(sb), diag=[(int((f.Borough == b).sum()), "You counted all event types in the borough. Add the EventType condition to COUNTIFS.")])],
            {"type": "excel", "steps": [f"=COUNTIFS(B:B,\"Shooting Permit\",H:H,\"{b}\")"]})
        avg = sb.dur_h.mean()
        add("film", ["dates", "calculated fields"], f"What is the average duration in hours of shooting permits in {b}? Round to one decimal.", 105,
            [dict(label="Average hours", answer=num(avg, 1), tol=0.06, diag=[
                (num(avg / 24, 3), "This looks like a value in days. Excel stores date-times as days: multiply (End − Start) by 24."),
                (num(f[f.Borough == b].pipe(lambda d: (d.EndDateTime - d.StartDateTime).dt.total_seconds().mean() / 3600), 1), "All event types are included. Restrict to shooting permits."),
            ])],
            {"type": "excel", "steps": ["Helper column: =(D2-C2)*24", f"=AVERAGEIFS(helper,B:B,\"Shooting Permit\",H:H,\"{b}\")"]})
        cnt = int(sb.ParkingHeld.str.contains("BROADWAY", case=False, na=False).sum())
        add("film", ["text search", "wildcards"], f"How many shooting permits in {b} have BROADWAY anywhere in ParkingHeld?", 75,
            [dict(label="Permits", answer=cnt, diag=[(0, "COUNTIFS without wildcards only matches cells that are exactly “BROADWAY”. Use \"*BROADWAY*\".")])],
            {"type": "excel", "steps": [f"=COUNTIFS(B:B,\"Shooting Permit\",H:H,\"{b}\",G:G,\"*BROADWAY*\")"]})
        cat = sb.Category.value_counts()
        add("film", ["pivot tables"], f"What is the most common Category of shooting permits in {b}, and how many permits does it have?", 75,
            [dict(label="Category", kind="text", answer=cat.index[0]), dict(label="Permits", answer=int(cat.iloc[0]), diag=[(len(sb), "This is the borough total. Count by Category.")])],
            {"type": "excel", "steps": ["PivotTable filtered on EventType and Borough, Category in Rows, Count in Values, sort descending."]})

    for y in [2013, 2014, 2015, 2016, 2017]:
        sy = shoot[shoot.StartDateTime.dt.year == y]
        add("film", ["dates", "COUNTIFS"], f"How many shooting permits started in {y}?", 75,
            [dict(label="Permits", answer=len(sy), diag=[(int((shoot.EnteredOn.dt.year == y).sum()), "You used the EnteredOn year. The question is about StartDateTime.")])],
            {"type": "excel", "steps": ["Helper column =YEAR(C2)", f"=COUNTIFS(B:B,\"Shooting Permit\",helper,{y})"]})
        lead = sy.lead_d.mean()
        add("film", ["dates", "calculated fields"], f"What is the average lead time in days between EnteredOn and StartDateTime for shooting permits starting in {y}? Round to one decimal.", 105,
            [dict(label="Average lead time (days)", answer=num(lead, 1), tol=0.06, diag=[
                (num(-lead, 1), "The sign is reversed. Calculate StartDateTime − EnteredOn."),
                (num(lead * 24, 1), "This looks like hours. Subtracting date-times in Excel already gives days."),
            ])],
            {"type": "excel", "steps": ["Helper column =C2-E2", f"=AVERAGEIFS(helper,B:B,\"Shooting Permit\",yearHelper,{y})"]})
    wd = shoot.StartDateTime.dt.day_name().value_counts()
    add("film", ["dates", "TEXT function"], "On which day of the week do most shooting permits start, and how many start on that day?", 90,
        [dict(label="Day", kind="text", answer=wd.index[0]), dict(label="Permits", answer=int(wd.iloc[0]))],
        {"type": "excel", "steps": ["Helper column =TEXT(C2,\"dddd\")", "PivotTable on the helper column, filtered on Shooting Permit."]})
    multi = shoot["CommunityBoard(s)"].fillna("").str.contains(",")
    add("film", ["text search", "percentages"], "What percentage of shooting permits required approval by more than one Community Board? Two decimals.", 120,
        [dict(label="Share (%)", kind="percent", answer=multi.mean(), diag=[
            (multi.sum() / len(f), "The denominator includes all event types. Divide by the number of shooting permits."),
        ])],
        {"type": "excel", "steps": ["Several boards are separated by commas: =COUNTIFS(B:B,\"Shooting Permit\",I:I,\"*,*\")", "Divide by the number of shooting permits."]})
    yrs = shoot.groupby([shoot.StartDateTime.dt.year, "Borough"]).size().unstack()
    first = int(yrs[yrs.Brooklyn > yrs.Manhattan].index.min())
    add("film", ["pivot tables"], "In which year did the number of shooting permits in Brooklyn first exceed those in Manhattan?", 90,
        [dict(label="Year", answer=first)],
        {"type": "excel", "steps": ["PivotTable: YEAR(StartDateTime) in Rows, Borough in Columns, count of EventID."]})
    student = shoot[shoot.Category == "Student"]
    add("film", ["COUNTIFS"], "How many shooting permits were granted for student films (Category = Student)?", 45,
        [dict(label="Permits", answer=len(student), diag=[(int((f.Category == "Student").sum()), "All event types are included. Restrict to shooting permits.")])],
        {"type": "excel", "steps": ["=COUNTIFS(B:B,\"Shooting Permit\",K:K,\"Student\")"]})


# ============================================================================= Excel: ETL cases
def build_etl():
    import csv
    with open(P["etl1"], newline="", encoding="utf-8-sig") as fh:
        recs = list(csv.reader(fh))
    header, body = recs[0], recs[1:]
    shifted = sum(1 for r in body if len(r) == len(header) + 1)
    # "Apple, Inc." is not wrapped in a text qualifier, so its rows split into one extra column
    body = [[r[0], r[1] + "," + r[2]] + r[3:] if len(r) == len(header) + 1 else r for r in body]
    raw = pd.DataFrame(body, columns=header)

    def money(v):
        v = v.strip()
        if v in ("", "-"):
            return 0.0
        neg = v.startswith("(")
        v = v.strip("()").replace(",", "")
        mult = 1
        if v.endswith(" M"):
            mult, v = 1e6, v[:-2]
        elif v.endswith(" K"):
            mult, v = 1e3, v[:-2]
        x = float(v) * mult
        return -x if neg else x
    add("etl1", ["ETL", "delimiters"], "Import the ETL case 1 file as comma-delimited. How many data rows end up with one column too many?", 60,
        [dict(label="Rows", answer=shifted, diag=[(0, "Look at Apple: “Apple, Inc.” has no text qualifier (quotes), so its comma splits the name into two columns.")])],
        {"type": "excel", "steps": ["Open the file in a text editor or import it with Data ▸ From Text/CSV.",
                                    "The Apple rows contain Apple, Inc. without quotes, so the comma acts as a delimiter and shifts every later value one column to the right.",
                                    "Fix: merge the two name columns for those rows (=B2&\",\"&C2) and shift the values back, or add quotes in the text file before importing."]})
    sol = {"type": "excel", "steps": ["Import with Data ▸ From Text/CSV. “Apple, Inc.” has no text qualifier, so fix the shifted Apple rows first.",
                                      "Convert “M”/“K” suffixes: =IF(RIGHT(D2,1)=\"M\",VALUE(LEFT(D2,LEN(D2)-2))*10^6, …).",
                                      "Parentheses mean negative: =IF(LEFT(H2,1)=\"(\",-VALUE(MID(H2,2,LEN(H2)-2)),VALUE(H2)).",
                                      "Remove spaces from tickers with SUBSTITUTE(A2,\" \",\"\"), fill missing tickers, remove the duplicate row."]}
    add("etl1", ["ETL", "data quality"], "How many rows in the ETL case 1 file are exact duplicates of an earlier row?", 60,
        [dict(label="Duplicate rows", answer=int(raw.duplicated().sum()))], sol)
    add("etl1", ["ETL", "data quality"], "How many rows have a missing Ticker?", 45,
        [dict(label="Rows", answer=int((raw.Ticker.str.strip() == "").sum()))], sol)
    tick = raw.Ticker.str.replace(" ", "")
    add("etl1", ["ETL", "SUBSTITUTE"], "After removing spaces from the tickers, how many distinct non-blank tickers are there?", 75,
        [dict(label="Distinct tickers", answer=int(tick[tick != ""].nunique()), diag=[
            (int(raw.Ticker[raw.Ticker.str.strip() != ""].nunique()), "Spaces are still in the tickers (A M Z N, AM ZN). Use SUBSTITUTE(A2,\" \",\"\")."),
        ])], sol)
    for name, year, col in [("Apple, Inc.", "2016", "Cash"), ("Aehr Test Systems", "2017", "Cash"), ("Aptevo Therapeutics Inc.", "2018", "Cash")]:
        v = raw[(raw.Name == name) & (raw.Year == year)][col].iloc[0]
        add("etl1", ["ETL", "units"], f"After cleaning, what is {name}’s {year} {col} in dollars?", 75,
            [dict(label=f"{col} ($)", answer=money(v), diag=[
                (float(v.split()[0]), "The value has a unit suffix (M = million, K = thousand). Multiply before using it."),
            ])], sol)
    v = raw[(raw.Ticker == "AAL") & (raw.Year == "2018")]["Total Depreciation"].iloc[0]
    add("etl1", ["ETL", "negative numbers"], "After cleaning, what is American Airlines’ 2018 Total Depreciation as a number?", 60,
        [dict(label="Total Depreciation", answer=money(v), diag=[(-money(v), "Parentheses mean a negative number. Keep the minus sign.")])], sol)
    add("etl1", ["ETL", "units"], "Compare Total Current Assets with Cash for American Airlines. In which unit is Total Current Assets reported?", 60,
        [dict(label="Unit", kind="choice", options=["Dollars", "Thousands of dollars", "Millions of dollars", "Billions of dollars"], answer="Millions of dollars",
              diag=[("Dollars", "AAL’s 2016 cash alone is 3,220,000,000 dollars, so 10,300 cannot be dollars."),
                    ("Thousands of dollars", "10,300 thousand = $10.3 million, less than the $3.2 billion in cash. Current assets include cash.")])], sol)

    # case 2
    c2 = pd.read_excel(P["etl2"], sheet_name="Case 2", dtype=str)
    m = c2.EmployeeCode.str.extract(r"^([A-Z]{3})(\d+)-(\d+)([WM])$")
    c2["Loc"], c2["Emp"], c2["Plant"], c2["Pay"] = m[0], m[1], m[2].astype(int), m[3]
    c2["LP"] = c2.Loc + "-" + c2.Plant.astype(str)
    s2 = {"type": "excel", "steps": ["Location: =LEFT(A2,3)", "EmpID: =MID(A2,4,4)", "PlantID: =MID(A2,FIND(\"-\",A2)+1,2)", "PayPeriod: =RIGHT(A2,1)", "LocationPlantID: =K2&\"-\"&L2"]}
    for loc in ["USA", "GER", "JAP", "MEX"]:
        add("etl2", ["text functions", "COUNTIF"], f"How many employees work in location {loc}?", 60,
            [dict(label="Employees", answer=int((c2.Loc == loc).sum()))], s2)
        sub = c2[c2.Loc == loc]
        add("etl2", ["text functions", "COUNTIFS"], f"How many employees in {loc} are paid monthly (M)?", 75,
            [dict(label="Monthly employees", answer=int((sub.Pay == "M").sum()), diag=[(int((c2.Pay == "M").sum()), "This covers all locations. Add the location condition.")])], s2)
    add("etl2", ["unique identifiers"], "How many distinct LocationPlantID values (e.g. USA-12) are there?", 90,
        [dict(label="Distinct LocationPlantIDs", answer=int(c2.LP.nunique()), diag=[
            (int(c2.Plant.nunique()), "This is the number of distinct plant numbers. Plant numbers repeat across countries, so you need Location + PlantID as the identifier."),
        ])], s2, why="Why is PlantID on its own not a unique identifier for a plant?")
    top = c2.LP.value_counts()
    add("etl2", ["unique identifiers", "pivot tables"], "Which LocationPlantID has the most employees, and how many?", 90,
        [dict(label="LocationPlantID", kind="text", answer=top.index[0]), dict(label="Employees", answer=int(top.iloc[0]))], s2)

    # case 3
    c3 = pd.read_excel(P["etl3"], sheet_name="Case 3 data", dtype=str)
    m = c3.EmployeeCode.str.extract(r"^([A-Za-z]+?)(\d+)-(\d+)([A-Za-z]+)$")
    c3["Loc"], c3["Emp"], c3["Plant"], c3["Pay"] = m[0], m[1].astype(int), m[2].astype(int), m[3]
    c3["LP"] = c3.Loc + "-" + c3.Plant.astype(str)
    s3 = {"type": "excel", "steps": ["Find where digits start: =MIN(FIND({0,1,2,3,4,5,6,7,8,9},A2&\"0123456789\"))",
                                     "Location: =LEFT(A2,start-1)", "PlantID: =VALUE(MID(A2,FIND(\"-\",A2)+1,2)) with an IF/ISNUMBER check for one-digit plants",
                                     "PayPeriod: IF(ISNUMBER(VALUE(MID(…))) …) or normalise with =IF(LEFT(suffix,1)=\"M\",\"Monthly\",\"Weekly\")"]}
    monthly = c3.Pay.str.startswith("M")
    add("etl3", ["text functions", "data cleaning"], "How many employees are paid monthly (codes M or Mo)?", 120,
        [dict(label="Monthly employees", answer=int(monthly.sum()), diag=[
            (int((c3.Pay == "M").sum()), "You only counted the code M. Mo means monthly too."),
            (int(c3.EmployeeCode.str.endswith("M").sum()), "RIGHT(code,1)=\"M\" misses the Mo codes, which end in o."),
        ])], s3)
    add("etl3", ["text functions", "VALUE"], "How many employees have a PlantID of 10 or higher?", 120,
        [dict(label="Employees", answer=int((c3.Plant >= 10).sum()), diag=[
            (len(c3), "Every row was counted. Your PlantID is probably text: in Excel, text is always larger than a number. Wrap the extraction in VALUE()."),
            (int(c3.EmployeeCode.str.extract(r"-(\d)")[0].astype(int).ge(10).sum()), "You only extracted one digit after the dash."),
        ])], s3)
    for loc in ["US", "Japan", "Canada", "GER"]:
        add("etl3", ["text functions"], f"How many employees have the location code {loc}?", 90,
            [dict(label="Employees", answer=int((c3.Loc == loc).sum()), diag=[
                (int(c3.EmployeeCode.str[:3].eq(loc[:3]).sum()) if len(loc) > 3 else None,
                 "Location codes in case 3 do not all have three letters. Extract everything before the first digit."),
            ])], s3)
    add("etl3", ["unique identifiers"], "How many distinct LocationPlantID values (e.g. US-3) are there?", 120,
        [dict(label="Distinct LocationPlantIDs", answer=int(c3.LP.nunique()), diag=[(int(c3.Plant.nunique()), "This is the number of distinct plant numbers. Combine Location and PlantID.")])], s3)

    # case 4
    x = pd.read_excel(P["etl4"], sheet_name=None)
    je, bu, coa, prep, src = x["JELineItems"], x["BusinessUnits"], x["ChartOfAccounts"], x["PreparerInfo"], x["Source"].dropna()
    for c in ["Debit", "Credit"]:
        je[c] = pd.to_numeric(je[c], errors="coerce").fillna(0)
    prep["Name"] = prep.PreparerName.str.split("|").str[0].str.replace("\xa0", " ").str.strip()
    j = je.merge(bu, on="BusinessUnitID", how="left").merge(coa[["GLAccountNumber", "GLAccountName", "AccountType", "AccountClass"]], on="GLAccountNumber", how="left")
    j = j.merge(prep[["PreparerID", "BusinessUnitID", "Name"]], on=["PreparerID", "BusinessUnitID"], how="left")
    s4 = {"type": "excel", "steps": ["BusinessUnit: =INDEX(BusinessUnits!B:B,MATCH(C2,BusinessUnits!A:A,0))",
                                     "GLAccountName / AccountType: INDEX/MATCH on GLAccountNumber (the number is not in the first column, so VLOOKUP fails).",
                                     "PreparerName needs both keys: =INDEX(PreparerInfo!A:A,MATCH(1,(PreparerInfo!B:B=D2)*(PreparerInfo!C:C=C2),0)) entered as an array formula.",
                                     "Then SUMIFS/COUNTIFS or a PivotTable on the flat file."]}
    for b in ["FOOD SERVICE", "HOTEL", "HOUSEKEEPING", "MAINTENANCE"]:
        add("etl4", ["INDEX/MATCH", "SUMIFS"], f"After joining the business units, what is the total Debit for business unit {b}?", 120,
            [dict(label="Total Debit", answer=num(j[j.BusinessUnit == b].Debit.sum()), diag=[
                (num(j[j.BusinessUnit == b].Amount.sum()), "You summed Amount (debits and credits net). The question asks for the Debit column."),
            ])], s4)
    for t in j.AccountType.dropna().unique()[:5]:
        add("etl4", ["INDEX/MATCH", "SUMIFS"], f"After joining the chart of accounts, what is the net Amount for AccountType “{t}”?", 120,
            [dict(label="Net Amount", answer=num(j[j.AccountType == t].Amount.sum()), diag=[
                (num(j[j.AccountType == t].Debit.sum()), "You summed only debits. Use the net Amount column."),
            ])], s4)
    wrong = je.merge(prep.drop_duplicates("PreparerID")[["PreparerID", "Name"]], on="PreparerID", how="left")
    for name in ["Carolyn Slater", "Alexander Knox", "Elizabeth Rees", "Brian Gibson", "Gavin McGrath"]:
        right = int((j.Name == name).sum())
        add("etl4", ["INDEX/MATCH", "unique identifiers"], f"How many journal lines were prepared by {name}?", 150,
            [dict(label="Journal lines", answer=right, diag=[
                (int((wrong.Name == name).sum()), "You matched on PreparerID only. IDs restart in each business unit, so match on PreparerID and BusinessUnitID together."),
                (int((je.PreparerID == prep[prep.Name == name].PreparerID.iloc[0]).sum()), "This counts every line with that PreparerID across all business units. Match on both keys."),
            ])], s4, why="Why does a plain VLOOKUP on PreparerID give the wrong preparer?")
    add("etl4", ["data quality"], "Check the journal: what is the sum of the Amount column for all lines (two decimals)?", 45,
        [dict(label="Sum of Amount", answer=num(je.Amount.sum()), tol=0.006)], s4)


# ============================================================================= Excel: IntegrateCo payroll
def build_payroll():
    p15 = pd.read_csv(P["pay15"], sep="|", dtype={"Job": str})
    p16 = pd.read_csv(P["pay16"], sep="|", dtype={"Project": str})
    jobs = pd.read_csv(P["jobs"], sep="|", dtype=str)
    locs = pd.read_csv(P["locs"], sep="|", dtype=str)
    rx = r"^\d{2}[HS]?(\d{3})"
    p15["code"] = p15.Job.str.extract(rx)[0]
    p16["code"] = p16.Project.str.extract(rx)[0]
    p15["gross"] = p15.Reg_Pay + p15.OT_Pay
    p15["right3"] = p15.Job.str[-3:]
    p16["right3"] = p16.Project.str[-3:]
    sol = {"type": "excel", "steps": ["Import with Data ▸ From Text/CSV and choose | as the delimiter. Import Job/Project as text.",
                                      "Job code: strip the 2-digit year and an optional H/S: =IF(ISNUMBER(VALUE(MID(H2,3,1))),MID(H2,3,3),MID(H2,4,3)).",
                                      "Do not use RIGHT(Job,3): location codes (B, R, SG) come after the job code.",
                                      "2015 gross pay = Reg_Pay + OT_Pay; 2016 gross pay = Gross_Wages. Then SUMIFS / PivotTables."]}
    g15, g16 = p15.gross.sum(), p16.Gross_Wages.sum()
    add("pay", ["SUM", "ETL"], "What is total gross pay (Reg_Pay + OT_Pay) in the first six months of 2015? Two decimals.", 90,
        [dict(label="2015 gross pay ($)", answer=num(g15), diag=[(num(p15.Reg_Pay.sum()), "You only summed Reg_Pay. Gross pay includes OT_Pay.")])], sol)
    add("pay", ["SUM", "ETL"], "By how much did gross pay increase from the first half of 2015 to the first half of 2016? Two decimals (positive = 2016 higher).", 120,
        [dict(label="Increase ($)", answer=num(g16 - g15), diag=[
            (num(g15 - g16), "The sign is reversed: compute 2016 − 2015."),
            (num(g16 - p15.Reg_Pay.sum()), "2015 OT_Pay is missing. 2016 Gross_Wages includes overtime, so add Reg_Pay + OT_Pay for 2015."),
        ])], sol, why="What has to be done to the 2015 data before it can be compared with 2016?")
    for code in ["001", "002", "005", "006", "010"]:
        desc = jobs[jobs.Job_Code == code].Job_Description.iloc[0]
        a15 = p15[p15.code == code].gross.sum()
        add("pay", ["text functions", "SUMIFS"], f"What was 2015 gross pay for job code {code} ({desc})? Two decimals.", 150,
            [dict(label="Gross pay ($)", answer=num(a15), diag=[
                (num(p15[p15.right3 == code].gross.sum()), "You used RIGHT(Job,3). That misses jobs with a location code after the job code (e.g. 14H001SG)."),
                (num(p15[p15.code == code].Reg_Pay.sum()), "OT_Pay is missing from gross pay."),
            ])], sol)
        a16 = p16[p16.code == code].Gross_Wages.sum()
        add("pay", ["text functions", "SUMIFS"], f"What was 2016 gross pay for job code {code} ({desc})? Two decimals.", 120,
            [dict(label="Gross pay ($)", answer=num(a16), diag=[
                (num(p16[p16.right3 == code].Gross_Wages.sum()), "You used RIGHT(Project,3). That misses projects with a location code after the job code."),
            ])], sol)
    ph = p16[p16.Employee_Number.isna()]
    add("pay", ["data quality", "COUNTBLANK"], "How many 2016 rows have no employee number (“phantom” entries), and what are their total gross wages?", 90,
        [dict(label="Phantom rows", answer=len(ph)), dict(label="Gross wages ($)", answer=num(ph.Gross_Wages.sum()))], sol)
    ot = p15.groupby(["Emp_No", "First_Name", "Last_Name"]).OT_Pay.sum().sort_values(ascending=False)
    (en, fn, ln), v = ot.index[0], ot.iloc[0]
    add("pay", ["pivot tables"], "Which employee earned the most overtime pay in 2015 and how much? Give the name as “First Last”.", 120,
        [dict(label="Employee", kind="text", answer=f"{fn} {ln}"), dict(label="OT pay ($)", answer=num(v), diag=[(num(p15.OT_Pay.max()), "This is the largest single row. Sum OT_Pay per employee first.")])], sol)
    cost = p16[["Gross_Wages", "SocSec_Medicare", "FUTA", "SUTA", "Work_Comp"]].sum().sum()
    add("pay", ["SUM"], "What was the total employee cost in 2016 (wages + Social Security/Medicare + FUTA + SUTA + workers’ compensation)? Two decimals.", 75,
        [dict(label="Total cost ($)", answer=num(cost), diag=[(num(g16), "This is wages only. Add the four employer costs.")])], sol)
    p15["date"] = pd.to_datetime(p15.Pay_Date, format="%m/%d/%Y")
    mth = p15.groupby(p15.date.dt.month).gross.sum()
    add("pay", ["dates", "pivot tables"], "In 2015, which month (number 1–6) had the highest gross wages, and how far above the six-month monthly average was it? Two decimals.", 150,
        [dict(label="Month", answer=int(mth.idxmax())), dict(label="Above average ($)", answer=num(mth.max() - mth.mean()), diag=[(num(mth.max()), "This is the month total. Subtract the average of the six monthly totals.")])], sol)
    per = p15.groupby(["First_Name", "Last_Name", "Pay_Date"])[["Reg_Hrs", "OT_Hrs"]].sum()
    n_ot = int(((per.Reg_Hrs < 40) & (per.OT_Hrs > 0)).sum())
    add("pay", ["pivot tables", "internal controls"], "2015: how many employee–pay date combinations show overtime hours although fewer than 40 regular hours were worked on that pay date?", 180,
        [dict(label="Combinations", answer=n_ot, diag=[(int(((p15.Reg_Hrs < 40) & (p15.OT_Hrs > 0)).sum()), "You tested individual rows. One pay date is split over several job codes: sum the hours per employee and pay date first.")])], sol)
    p15["name"] = p15.First_Name + " " + p15.Last_Name
    l15 = p15.merge(locs, left_on="name", right_on="Employee_Name", how="left")
    l16 = p16.dropna(subset=["Employee_Name"]).merge(locs, on="Employee_Name", how="left")
    for loc in ["1", "2"]:
        a = l16[l16.Location == loc]
        rate = a.Gross_Wages.sum() / (a.Hours + a.OT_Hours).sum()
        add("pay", ["INDEX/MATCH", "calculated fields"], f"Using the locations file, what was the 2016 average gross pay per hour (regular + overtime hours) at location {loc}? Two decimals.", 180,
            [dict(label="Pay per hour ($)", answer=num(rate), diag=[(num(a.Gross_Wages.sum() / a.Hours.sum()), "Overtime hours are missing from the denominator.")])], sol)
        b = l15[l15.Location == loc]
        add("pay", ["INDEX/MATCH", "SUMIFS"], f"Using the locations file, what were 2015 gross wages at location {loc}? Two decimals.", 150,
            [dict(label="Gross wages ($)", answer=num(b.gross.sum()), diag=[(num(b.Reg_Pay.sum()), "OT_Pay is missing.")])], sol)


# ============================================================================= Regression (data interpretation)
def build_regression():
    s = pd.read_excel(P["rest"], sheet_name="salesData")
    w = pd.read_excel(P["rest"], sheet_name="weatherData")
    g = pd.read_excel(P["rest"], sheet_name="gameData")
    daily = s.groupby("date")[["netTotal", "netTotal_364", "netTotal_7", "netTotal_28"]].sum().reset_index()
    d = daily.merge(w, on="date", how="left").merge(g[["date", "gameDay", "homeGame"]], on="date", how="left")
    d[["gameDay", "homeGame"]] = d[["gameDay", "homeGame"]].fillna(0)
    d["weekend"] = (d.date.dt.dayofweek >= 5).astype(int)
    d = d.dropna(subset=["tmax"])

    def ols(y, X, names):
        Xm = np.column_stack([np.ones(len(X))] + [X[c].values.astype(float) for c in X.columns])
        yv = y.values.astype(float)
        beta, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
        resid = yv - Xm @ beta
        n, k = Xm.shape
        s2 = resid @ resid / (n - k)
        se = np.sqrt(np.diag(s2 * np.linalg.inv(Xm.T @ Xm)))
        t = beta / se
        p = 2 * stats.t.sf(np.abs(t), n - k)
        r2 = 1 - (resid @ resid) / ((yv - yv.mean()) @ (yv - yv.mean()))
        # coefficients are stored as displayed (4 decimals) so a student's hand calculation matches the key
        return dict(n=int(n), r2=float(r2), coefs=[dict(term=nm, b=round(float(b), 4), se=round(float(e), 4), t=round(float(tt), 3), p=float(pp))
                                                     for nm, b, e, tt, pp in zip(["Intercept"] + names, beta, se, t, p)])

    models = []
    d["sales364_x_game"] = d.netTotal_364 * d.gameDay
    specs = [
        ("M1", "Daily sales and temperature", ["tmax"], {"tmax": "Maximum temperature (°F)"}),
        ("M2", "Daily sales on game days", ["gameDay"], {"gameDay": "Game day (1 = game, 0 = no game)"}),
        ("M3", "Last year’s sales and game days", ["netTotal_364", "gameDay", "sales364_x_game"],
         {"netTotal_364": "Sales 364 days earlier ($)", "gameDay": "Game day (1/0)", "sales364_x_game": "Sales 364 days earlier × Game day"}),
        ("M4", "Forecasting model", ["netTotal_7", "netTotal_364", "tmax", "gameDay"],
         {"netTotal_7": "Sales 7 days earlier ($)", "netTotal_364": "Sales 364 days earlier ($)", "tmax": "Maximum temperature (°F)", "gameDay": "Game day (1/0)"}),
        ("M5", "Weather and sales", ["prcp", "snow", "tmax"], {"prcp": "Precipitation (inches)", "snow": "Snow (inches)", "tmax": "Maximum temperature (°F)"}),
        ("M6", "Weekends and home games", ["weekend", "homeGame"], {"weekend": "Weekend (1 = Sat/Sun)", "homeGame": "Home game (1/0)"}),
    ]
    for mid, title, xs, labels in specs:
        dm = d.dropna(subset=xs + ["netTotal"])
        r = ols(dm.netTotal, dm[xs], xs)
        for c in r["coefs"]:
            c["label"] = labels.get(c["term"], "Intercept")
        models.append(dict(id=mid, title=title, y="Daily sales, all menu items ($)", **r))

    # product-level price models
    for item in s.LineItem.value_counts().index[:8]:
        sub = s[s.LineItem == item].dropna(subset=["quantity", "price", "discounts"])
        cols = [c for c in ["price", "discounts"] if sub[c].std() > 0]
        if "price" not in cols:
            continue
        r = ols(sub.quantity, sub[cols], cols)
        lab = {"Intercept": "Intercept", "price": "Average price ($)", "discounts": "Discounts granted ($)"}
        for c in r["coefs"]:
            c["label"] = lab[c["term"]]
        models.append(dict(id=f"P{len(models) - 5}", title=f"Demand for {item}", y=f"Daily quantity of {item} sold", **r))

    rng = random.Random(7)
    exams = []
    for i in range(30):
        m = models[i % len(models)]
        terms = m["coefs"]
        qs = []
        # Q1: prediction
        inputs = {}
        for c in terms[1:]:
            t = c["term"]
            col = {"sales364_x_game": None}.get(t, t)
            if t == "sales364_x_game":
                continue
            if t in ("gameDay", "homeGame", "weekend"):
                inputs[t] = rng.choice([0, 1])
            elif m["id"].startswith("P"):
                src = s[s.LineItem == m["title"].replace("Demand for ", "")][t]
                inputs[t] = int(round(rng.uniform(src.quantile(.2), src.quantile(.8))))
            else:
                inputs[t] = int(round(rng.uniform(d[t].quantile(.2), d[t].quantile(.8)), -1 if d[t].mean() > 100 else 0))
        if any(c["term"] == "sales364_x_game" for c in terms):
            inputs["gameDay"] = 1
            inputs["sales364_x_game"] = inputs["netTotal_364"] * 1
        pred = terms[0]["b"] + sum(c["b"] * inputs[c["term"]] for c in terms[1:])
        no_inter = pred - sum(c["b"] * inputs[c["term"]] for c in terms[1:] if c["term"] == "sales364_x_game")
        shown_inputs = {terms_lbl(m, k): v for k, v in inputs.items() if k != "sales364_x_game"}
        qs.append(dict(kind="predict", prompt="Use the model to predict the outcome for these values: " +
                       "; ".join(f"{k} = {v:g}" for k, v in shown_inputs.items()) + ". Round to two decimals.",
                       fields=[dict(label="Prediction", kind="number", answer=num(pred), tol=max(0.6, abs(pred) * 0.002),
                                    diagnostics=[x for x in [
                                        {"value": num(no_inter), "msg": "The interaction term is missing. For a game day, add b × (Sales 364 × 1)."} if abs(no_inter - pred) > 1 else None,
                                        {"value": num(pred - terms[0]["b"]), "msg": "The intercept is missing."} if abs(terms[0]["b"]) > 1 else None,
                                    ] if x])],
                       solution=f"Prediction = {terms[0]['b']:g}" + "".join(f" + ({c['b']:g} × {inputs[c['term']]:g})" for c in terms[1:]) + f" = {pred:.2f}"))
        # Q2: significance
        c = terms[1 + (i // len(models)) % (len(terms) - 1)]
        sig = c["p"] < 0.05
        qs.append(dict(kind="significance", prompt=f"Is the coefficient on “{c['label']}” statistically significant at the 5% level?",
                       fields=[dict(label="Significant at 5%?", kind="choice", options=["Yes", "No"], answer="Yes" if sig else "No",
                                    diagnostics=[{"value": "No" if sig else "Yes",
                                                  "msg": f"The p-value is {c['p']:.3f}. Significant means p < 0.05. A large coefficient is not the same as a significant one."}])],
                       solution=f"p = {c['p']:.3f} {'<' if sig else '≥'} 0.05, so the coefficient is {'' if sig else 'not '}statistically significant at the 5% level."))
        # Q3: interpretation (multiple choice with mistake-based distractors)
        dummy = c["term"] in ("gameDay", "homeGame", "weekend")
        sales = m["y"].startswith("Daily sales")
        ytxt = "predicted daily sales" if sales else "predicted daily quantity sold"
        amt = (lambda v: f"${abs(v):,.2f}") if sales else (lambda v: f"{abs(v):.2f} units")
        dirw = "higher" if c["b"] > 0 else "lower"
        inter = c["term"] == "sales364_x_game"
        ctrl = "Holding the other variables constant, " if len(terms) > 2 else ""
        name = lc(re.sub(r"\s*\(.*?\)", "", c["label"]))
        if inter:
            correct = (f"On game days, the effect of $1 more sales 364 days earlier on predicted daily sales is {abs(c['b']):.4f} {dirw} "
                       f"than on other days: the slope is {terms[1]['b']:.4f} + ({c['b']:.4f}) = {terms[1]['b'] + c['b']:.4f}.")
            wrong = [f"Game days cause daily sales to change by ${abs(c['b']):.4f}.",
                     f"Each extra $1 of sales 364 days earlier changes predicted daily sales by {c['b']:.4f} on every day.",
                     f"Predicted daily sales on a game day are ${abs(c['b']):.4f}."]
        elif dummy:
            correct = f"{ctrl}{ytxt} is {amt(c['b'])} {dirw} when {name} = 1 than when {name} = 0."
            correct = correct[0].upper() + correct[1:]
            wrong = [f"A {name} causes daily sales to be {amt(c['b'])} {dirw}.",
                     f"{ytxt[0].upper() + ytxt[1:]} is {abs(c['b']):.2f}% {dirw} when {name} = 1.",
                     f"{ytxt[0].upper() + ytxt[1:]} when {name} = 1 is {amt(c['b'])}."]
        else:
            phrase = {"tmax": "a 1 °F higher maximum temperature", "prcp": "one more inch of precipitation", "snow": "one more inch of snow",
                      "price": "a $1 higher average price", "discounts": "$1 more in discounts", "netTotal_7": "$1 more sales 7 days earlier",
                      "netTotal_364": "$1 more sales 364 days earlier"}.get(c["term"], f"one more unit of {name}")
            correct = f"{ctrl}{phrase} is associated with {amt(c['b'])} {dirw} {ytxt}."
            correct = correct[0].upper() + correct[1:]
            cap = phrase[0].upper() + phrase[1:]
            wrong = [f"{cap} causes {amt(c['b'])} {dirw} {ytxt[10:]}.",
                     f"{cap} is associated with {abs(c['b']):.2f}% {dirw} {ytxt}.",
                     f"{ytxt[0].upper() + ytxt[1:]} is {amt(c['b'])} when {name} is zero."]
        fix = lambda t: t.replace("daily sales is", "daily sales are")
        correct, wrong = fix(correct), [fix(w) for w in wrong]
        opts = [correct] + wrong
        rng.shuffle(opts)
        qs.append(dict(kind="interpret", prompt=f"Which statement correctly interprets the coefficient on “{c['label']}”?",
                       fields=[dict(label="Interpretation", kind="choice", options=opts, answer=correct,
                                    diagnostics=[{"value": wrong[0], "msg": "Regression on observational data shows an association, not a cause. Avoid “causes”."},
                                                 {"value": wrong[1], "msg": ("The coefficient is in the outcome’s own units, not percent." if not inter else "That ignores the interaction: the extra slope applies only on game days.")},
                                                 {"value": wrong[2], "msg": "That describes an intercept or a predicted level, not a coefficient."}])],
                       solution=correct))
        # Q4: R²
        r2 = m["r2"]
        right = f"About {r2 * 100:.0f}% of the variation in the outcome is explained by the model."
        opts = [right, f"The model’s predictions are correct {r2 * 100:.0f}% of the time.",
                f"The predictors cause {r2 * 100:.0f}% of the outcome.", f"The model has a {r2 * 100:.0f}% chance of being the right model."]
        rng.shuffle(opts)
        qs.append(dict(kind="r2", prompt=f"The model’s R² is {r2:.3f}. What does that mean?",
                       fields=[dict(label="Meaning of R²", kind="choice", options=opts, answer=right,
                                    diagnostics=[{"value": o, "msg": "R² is the share of the variation in the outcome that the model explains. It is not an accuracy rate or a probability, and it says nothing about causality."} for o in opts if o != right])],
                       solution=right + (" A low R² means a lot of the day-to-day variation is not captured, so individual predictions are uncertain." if r2 < 0.5 else " Even so, check whether the model would hold for new dates.")))
        exams.append(dict(id=f"R{i + 1:02d}", model=m["id"], questions=qs, seconds=300))
    return models, exams


def lc(t):
    return t[:1].lower() + t[1:]


def terms_lbl(m, term):
    for c in m["coefs"]:
        if c["term"] == term:
            return c["label"]
    return term


# ============================================================================= exams
def build_exams():
    rng = random.Random(2026)
    by_file = {}
    for q in QUESTIONS:
        by_file.setdefault(q["file"], []).append(q)
    excel_files = [f for f in by_file if FILES[f]["subject"] == "Excel"]
    sql_files = [f for f in by_file if FILES[f]["subject"] == "SQL"]
    exams = []
    seen = set()
    tries = 0
    while len(exams) < 300 and tries < 20000:
        tries += 1
        xf, sf = rng.choice(excel_files), rng.choice(sql_files)
        xs = rng.sample(by_file[xf], min(3, len(by_file[xf])))
        ss = rng.sample(by_file[sf], min(3, len(by_file[sf])))
        total = sum(q["seconds"] for q in xs + ss)
        if not 300 <= total <= 600:
            continue
        key = tuple(sorted(q["id"] for q in xs + ss))
        if key in seen:
            continue
        seen.add(key)
        exams.append(dict(id=f"E{len(exams) + 1:03d}", excel=xf, sql=sf, questionIds=[q["id"] for q in xs + ss], seconds=480))
    return exams


def main():
    print("Building P-cards …"); build_pcards()
    print("Building movies …"); build_movies()
    print("Building songs …"); build_songs()
    print("Building Chinook …"); build_chinook()
    print("Building film permits …"); build_film()
    print("Building ETL cases …"); build_etl()
    print("Building payroll …"); build_payroll()
    print("Building regression …"); models, rexams = build_regression()
    exams = build_exams()

    # every field must have a usable answer
    for q in QUESTIONS:
        for f in q["fields"]:
            a = f["answer"]
            assert a is not None and not (isinstance(a, float) and math.isnan(a)), (q["id"], q["prompt"])

    DS.mkdir(parents=True, exist_ok=True)
    downloads = {}
    for key, path in P.items():
        target = DS / path.name
        if not target.exists() or target.stat().st_size != path.stat().st_size:
            shutil.copy2(path, target)
        downloads[key] = dict(name=path.name, href=f"datasets/{path.name}", size=path.stat().st_size)

    n_diag = sum(len(f["diagnostics"]) for q in QUESTIONS for f in q["fields"])
    bank = dict(generated=pd.Timestamp.now().strftime("%Y-%m-%d"), files=FILES, downloads=downloads,
                questions=QUESTIONS, exams=exams, regression=dict(models=models, exams=rexams),
                stats=dict(questions=len(QUESTIONS), diagnostics=n_diag, exams=len(exams), regressionExams=len(rexams)))
    (SITE / "bank.js").write_text("window.BANK = " + json.dumps(bank, ensure_ascii=False, indent=None) + ";\n", encoding="utf-8")
    print(json.dumps(bank["stats"]))


if __name__ == "__main__":
    main()
