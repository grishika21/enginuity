"""Generate the synthetic sample data for the demo.

Everything is fictional: "Kestrel Industrial" and its KP-200 pump and KC-50 compressor do not
exist. Run once (`python scripts/make_samples.py`); `scripts/seed.py` then loads the samples.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data" / "samples"
LIBRARY = ROOT / "data" / "library"

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1x", parent=styles["Heading1"], fontSize=16, spaceBefore=10, spaceAfter=6)
H2 = ParagraphStyle("H2x", parent=styles["Heading2"], fontSize=13, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("Bodyx", parent=styles["BodyText"], fontSize=10, leading=14)
TITLE = ParagraphStyle("Titlex", parent=styles["Title"], fontSize=20)

PARTS = [
    ("KP-200", "KP-200-101", "Pump casing", 1),
    ("KP-200", "KP-200-110", "Shaft", 1),
    ("KP-200", "KP-200-111", "Shaft sleeve", 1),
    ("KP-200", "KP-200-114", "Mechanical seal assembly", 1),
    ("KP-200", "KP-200-114B", "Mechanical seal assembly, improved (SB-07)", 1),
    ("KP-200", "KP-200-115", "Casing O-ring, 210 x 4 mm", 1),
    ("KP-200", "KP-200-120", "Impeller, 245 mm", 1),
    ("KP-200", "KP-200-121", "Impeller nut", 1),
    ("KP-200", "KP-200-130", "Deep-groove ball bearing 6309", 2),
    ("KP-200", "KP-200-131", "Bearing lip seal", 2),
    ("KP-200", "KP-200-140", "Flexible coupling", 1),
    ("KP-200", "KP-200-150", "Pressure gauge, 0-16 bar", 1),
    ("KC-50", "KC-50-201", "Crankcase", 1),
    ("KC-50", "KC-50-210", "Intake air filter element", 1),
    ("KC-50", "KC-50-220", "Compressor oil, 1 L", 2),
    ("KC-50", "KC-50-230", "Valve plate assembly", 1),
    ("KC-50", "KC-50-240", "Drive belt, B-section", 2),
    ("KC-50", "KC-50-250", "Safety relief valve, 11 bar", 1),
    ("KC-50", "KC-50-260", "Motor bearing 6205", 2),
]


def steps(items: list[str]) -> list:
    return [Paragraph(f"{i}. {text}", BODY) for i, text in enumerate(items, 1)]


def table(rows: list[list[str]], widths: list[float]) -> Table:
    t = Table(rows, colWidths=[w * mm for w in widths])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde3ea")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def p(text: str) -> Paragraph:
    return Paragraph(text, BODY)


def kp200_manual(path: Path) -> None:
    s = [Paragraph("Kestrel KP-200 Centrifugal Pump", TITLE),
         p("Component Maintenance Manual, revision C. Kestrel Industrial (fictional demo data)."),
         Spacer(1, 6),
         Paragraph("1 General Description", H1),
         p("The KP-200 is an end-suction, single-stage centrifugal pump for clean water and light process "
           "fluids at up to 120 m3/h and 9 bar. A 22 kW four-pole motor drives the pump shaft through a "
           "flexible coupling (KP-200-140)."),
         Paragraph("1.1 Components", H2),
         p("The main components are the pump casing (KP-200-101), the impeller (KP-200-120) held by the "
           "impeller nut (KP-200-121), the shaft (KP-200-110) with its shaft sleeve (KP-200-111), the "
           "mechanical seal assembly (KP-200-114) and two deep-groove ball bearings (KP-200-130) in the "
           "bearing housing, each protected by a bearing lip seal (KP-200-131). The casing joint is sealed "
           "by the casing O-ring (KP-200-115)."),
         Paragraph("1.2 Controls and Indicating", H2),
         p("A pressure gauge (KP-200-150) on the discharge flange shows the delivery pressure. The motor "
           "starter has a local start/stop station and an emergency stop. A dry-run protection relay stops "
           "the motor when the suction pressure stays below 0.3 bar for more than 10 seconds."),
         Paragraph("1.3 Interfaces", H2),
         p("Suction flange DN100, discharge flange DN80, both PN16. The seal flush line is taken from the "
           "discharge side through a 6 mm orifice. The baseplate is grouted to the foundation."),
         Paragraph("1.4 Power Supply", H2),
         p("The motor is supplied at 415 V, 3-phase, 50 Hz through a star-delta starter rated 45 A. The "
           "dry-run relay is supplied at 24 V DC from the control panel."),
         Paragraph("2 Specifications", H1),
         table([["Item", "Value"], ["Rated flow", "120 m3/h"], ["Rated head", "62 m"], ["Speed", "1475 rpm"],
                ["Max casing pressure", "16 bar"], ["Impeller nut torque", "95 Nm"],
                ["Casing bolt torque", "55 Nm, cross pattern"], ["Bearing grease", "Lithium complex, NLGI 2"]],
               [60, 80]),
         Paragraph("3 Maintenance Schedule", H1),
         table([["Interval", "Task", "Section"], ["Weekly", "Check for seal leakage and noise", "5"],
                ["Every 2000 h", "Lubricate bearings (15 g each)", "4.3"],
                ["Every 8000 h", "Inspect impeller and wear clearance", "4.2"],
                ["On leakage", "Replace mechanical seal", "4.1"]], [30, 90, 20]),
         PageBreak(),
         Paragraph("4 Procedures", H1),
         Paragraph("4.1 Mechanical Seal Replacement", H2),
         p("Warning: isolate the motor electrically and lock out the starter. Close the suction and "
           "discharge valves and drain the casing before you start."),
         *steps([
             "Remove the coupling guard and disconnect the flexible coupling (KP-200-140).",
             "Remove the casing bolts and withdraw the back pull-out unit from the casing.",
             "Remove the impeller nut (KP-200-121) and pull the impeller (KP-200-120) off the shaft.",
             "Remove the seal cover and slide the old mechanical seal assembly (KP-200-114) off the shaft sleeve.",
             "Inspect the shaft sleeve (KP-200-111) for scoring. Replace the sleeve if the wear groove is deeper than 0.1 mm.",
             "Clean the seal chamber and lightly lubricate the new seal faces with clean water. Do not use grease on the faces.",
             "Fit the new mechanical seal assembly (KP-200-114) and set the seal working length to 42.5 mm.",
             "Refit the impeller and torque the impeller nut to 95 Nm.",
             "Fit a new casing O-ring (KP-200-115) and refit the back pull-out unit. Torque the casing bolts to 55 Nm in a cross pattern.",
             "Reconnect the coupling, check alignment (maximum 0.05 mm offset) and refit the guard.",
         ]),
         p("Test: open the valves, vent the casing and run the pump for 30 minutes. The seal must not leak "
           "more than 5 drops per minute after the first 10 minutes."),
         Paragraph("4.2 Impeller Inspection and Replacement", H2),
         *steps([
             "Isolate, drain and remove the back pull-out unit as in steps 1 and 2 of section 4.1.",
             "Measure the wear-ring clearance. Replace the impeller (KP-200-120) if the diametral clearance exceeds 0.9 mm.",
             "Check the vanes for cavitation pitting deeper than 2 mm.",
             "Fit the impeller with a new impeller nut (KP-200-121) and torque to 95 Nm.",
         ]),
         Paragraph("4.3 Bearing Lubrication", H2),
         p("Lubricate each bearing (KP-200-130) every 2000 operating hours with 15 g of lithium complex "
           "grease, NLGI 2, through the grease nipples while the pump runs. Do not over-grease: bearing "
           "temperature above 85 C after lubrication means too much grease."),
         Paragraph("5 Troubleshooting", H1),
         table([["Symptom", "Likely cause", "Action"],
                ["Seal leaks steadily", "Worn seal faces or sleeve", "Replace seal (4.1)"],
                ["Low delivery pressure", "Worn impeller, air in suction", "Inspect impeller (4.2), vent"],
                ["Hot bearings", "Over-greasing, misalignment", "Check grease quantity, align coupling"]],
               [45, 55, 50]),
         ]
    SimpleDocTemplate(str(path), pagesize=A4, title="KP-200 Component Maintenance Manual").build(s)


def kc50_manual(path: Path) -> None:
    s = [Paragraph("Kestrel KC-50 Piston Air Compressor", TITLE),
         p("Operation and Maintenance Manual, revision B. Kestrel Industrial (fictional demo data)."),
         Paragraph("1 General Description", H1),
         p("The KC-50 is a two-stage, belt-driven piston compressor delivering 50 m3/h of free air at up "
           "to 10 bar into a 270 L receiver. A 7.5 kW motor drives the crankcase (KC-50-201) through two "
           "drive belts (KC-50-240)."),
         Paragraph("1.1 Components", H2),
         p("Crankcase (KC-50-201), valve plate assembly (KC-50-230), intake air filter element (KC-50-210), "
           "safety relief valve (KC-50-250) set at 11 bar, and two motor bearings (KC-50-260)."),
         Paragraph("1.2 Controls and Indicating", H2),
         p("A pressure switch starts the compressor at 8 bar and stops it at 10 bar. An oil level sight "
           "glass is fitted on the crankcase."),
         Paragraph("1.3 Power Supply", H2),
         p("415 V, 3-phase, 50 Hz through a direct-on-line starter with thermal overload set at 15 A."),
         Paragraph("2 Maintenance Schedule", H1),
         table([["Interval", "Task", "Section"], ["Daily", "Drain receiver condensate", "3.4"],
                ["Every 500 h", "Clean or replace intake filter", "3.1"],
                ["Every 1000 h", "Change compressor oil", "3.2"],
                ["Every 4000 h", "Lubricate motor bearings", "3.3"]], [30, 90, 20]),
         Paragraph("3 Procedures", H1),
         Paragraph("3.1 Intake Filter Replacement", H2),
         *steps(["Stop the compressor and isolate the motor.", "Unscrew the filter housing cover.",
                 "Replace the filter element (KC-50-210) and refit the cover hand-tight."]),
         Paragraph("3.2 Oil Change", H2),
         *steps(["Run the compressor for 10 minutes to warm the oil, then isolate it.",
                 "Drain the oil through the crankcase drain plug.",
                 "Refill with 2 L of compressor oil (KC-50-220) to the middle of the sight glass."]),
         Paragraph("3.3 Motor Bearing Lubrication", H2),
         p("Lubricate the motor bearings (KC-50-260) every 4000 operating hours with 8 g of polyurea "
           "grease each. Bearings are lubricated with the motor stopped."),
         Paragraph("3.4 Condensate Drain", H2),
         p("Open the receiver drain valve daily until only air comes out."),
         ]
    SimpleDocTemplate(str(path), pagesize=A4, title="KC-50 Operation and Maintenance Manual").build(s)


def pi_0042(path: Path) -> None:
    sections: list[tuple[str, str, list[str]]] = [
        ("1 Scope", "", ["This process instruction covers the assembly of the KP-200 pump unit from "
                         "inspected components at the Kestrel assembly line, station A3."]),
        ("2 Safety", "", ["Wear safety shoes, gloves and eye protection.",
                          "Use the overhead hoist for the casing (78 kg). Never lift by the shaft."]),
        ("3 Tools and Fixtures", "", ["Bearing induction heater, set to 110 C.",
                                      "Torque wrench 20-200 Nm, calibrated.",
                                      "Seal setting gauge 42.5 mm.", "Dial indicator with magnetic base."]),
        ("4 Bearing Housing Sub-Assembly", "", []),
        ("4.1 Prepare the Housing", "", ["Clean the bearing housing bores and check them for burrs.",
                                         "Record the housing serial number on the route card."]),
        ("4.2 Fit the Bearings", "", ["Heat each bearing (KP-200-130) to 110 C on the induction heater.",
                                      "Slide the bearing onto the shaft (KP-200-110) against the shoulder.",
                                      "Let the bearings cool to room temperature before you continue."]),
        ("4.3 Fit the Lip Seals", "", ["Press a bearing lip seal (KP-200-131) into each end cover with the lip facing outward.",
                                       "Fit the end covers and torque the cover screws to 12 Nm."]),
        ("5 Rotating Assembly", "", []),
        ("5.1 Shaft Sleeve", "", ["Apply anti-seize compound to the shaft.",
                                  "Slide the shaft sleeve (KP-200-111) on until it meets the shoulder."]),
        ("5.2 Mechanical Seal", "", ["Set the mechanical seal assembly (KP-200-114) to a working length of 42.5 mm with the setting gauge.",
                                     "Lubricate the seal faces with clean water only.",
                                     "Slide the seal onto the sleeve and tighten the grub screws to 4 Nm."]),
        ("5.3 Impeller", "", ["Fit the impeller key and slide the impeller (KP-200-120) onto the shaft.",
                              "Fit the impeller nut (KP-200-121) and torque it to 95 Nm.",
                              "Check that the shaft turns freely by hand."]),
        ("5.4 Balance Check", "", ["Mount the rotating assembly on the balancing stand.",
                                   "The residual unbalance must be below grade G6.3. Record the reading."]),
        ("6 Casing and Final Assembly", "", []),
        ("6.1 Casing Closure", "", ["Fit a new casing O-ring (KP-200-115) in the casing groove.",
                                    "Lower the back pull-out unit into the casing (KP-200-101) with the hoist.",
                                    "Torque the casing bolts to 55 Nm in a cross pattern, in two passes."]),
        ("6.2 Coupling and Alignment", "", ["Fit the flexible coupling (KP-200-140) halves.",
                                            "Align the motor to within 0.05 mm offset and 0.05 mm/100 mm angularity.",
                                            "Fit the coupling guard."]),
        ("6.3 Instruments", "", ["Fit the pressure gauge (KP-200-150) on the discharge flange with PTFE tape."]),
        ("7 Testing", "", []),
        ("7.1 Hydrostatic Test", "", ["Fill the casing with water and pressurise to 24 bar (1.5 x max casing pressure).",
                                      "Hold for 10 minutes. No visible leakage is allowed."]),
        ("7.2 Performance Run", "", ["Run the pump at rated speed for 30 minutes on the test loop.",
                                     "Record flow, head, current and bearing temperatures every 10 minutes.",
                                     "Bearing temperature must stay below 80 C."]),
        ("8 Packing and Dispatch", "", ["Drain the pump and blow the casing dry with oil-free air.",
                                        "Fit flange covers, attach the test certificate and pack on a pallet."]),
    ]
    s = [Paragraph("Process Instruction PI-0042", TITLE),
         p("Assembly of the KP-200 Pump Unit. Kestrel Industrial, revision 4 (fictional demo data)."),
         Spacer(1, 6)]
    for title, _unused, lines in sections:
        top = "." not in title.split(" ")[0]
        s.append(Paragraph(title, H1 if top else H2))
        if lines:
            if title.startswith(("1 ", "2 ", "3 ")):
                s.extend(p(line) for line in lines)
            else:
                s.extend(steps(lines))
        if title.startswith("5.2") or title.startswith("6.3"):
            s.append(PageBreak())
    SimpleDocTemplate(str(path), pagesize=A4, title="PI-0042 KP-200 Pump Assembly").build(s)


def scanned_bulletin(path: Path) -> None:
    """An image-only PDF, like a scanned paper bulletin, to exercise the OCR path."""
    lines = [
        "KESTREL INDUSTRIAL - SERVICE BULLETIN SB-07",
        "",
        "Subject: KP-200 mechanical seal, improved design",
        "Applies to: KP-200 pumps, serial numbers 2200 to 2650",
        "",
        "Reason: early seal face wear has been reported on pumps",
        "running with abrasive water above 40 C.",
        "",
        "Action: when the mechanical seal is next replaced,",
        "fit the improved seal assembly KP-200-114B instead of",
        "KP-200-114. Set the working length to 43.0 mm.",
        "The casing O-ring KP-200-115 must also be replaced.",
        "",
        "Compliance: at the next seal replacement.",
        "",
        "Fictional demo document - not a real bulletin.",
    ]
    img = Image.new("L", (1240, 1754), color=250)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 34)
    except OSError:
        font = ImageFont.load_default()
    y = 160
    for line in lines:
        draw.text((110, y), line, fill=20, font=font)
        y += 62
    img = img.rotate(0.6, fillcolor=250).filter(ImageFilter.GaussianBlur(0.6))
    img.convert("RGB").save(path, "PDF", resolution=150)


LIBRARY_DOCS = [
    {"id": "TP-101", "title": "KP-200 Pump Overhaul", "equipment": "KP-200",
     "parts": ["KP-200-114", "KP-200-115", "KP-200-120", "KP-200-130"]},
    {"id": "TP-102", "title": "KP-200 Seasonal Shutdown and Restart", "equipment": "KP-200",
     "parts": ["KP-200-115", "KP-200-150"]},
    {"id": "TP-103", "title": "KP-200 Coupling Alignment", "equipment": "KP-200", "parts": ["KP-200-140"]},
    {"id": "TP-201", "title": "KC-50 Intake Filter Change", "equipment": "KC-50", "parts": ["KC-50-210"]},
]


def library() -> None:
    LIBRARY.mkdir(parents=True, exist_ok=True)
    graph = {"documents": {}}
    for doc in LIBRARY_DOCS:
        body = (f"---\nid: {doc['id']}\ntitle: {doc['title']}\nequipment: {doc['equipment']}\n"
                f"parts: [{', '.join(doc['parts'])}]\n---\n\n# {doc['title']}\n\n"
                f"Published procedure (demo seed). Parts: {', '.join(doc['parts'])}.\n")
        (LIBRARY / f"{doc['id']}.md").write_text(body, encoding="utf-8")
        graph["documents"][doc["id"]] = {"title": doc["title"], "equipment": doc["equipment"], "parts": doc["parts"],
                                         "file": f"{doc['id']}.md"}
    (LIBRARY / "graph.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")


def main() -> int:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    kp200_manual(SAMPLES / "KP-200_Maintenance_Manual.pdf")
    kc50_manual(SAMPLES / "KC-50_Maintenance_Manual.pdf")
    pi_0042(SAMPLES / "PI-0042_KP-200_Pump_Assembly.pdf")
    scanned_bulletin(SAMPLES / "SB-07_KP-200_Seal_Bulletin_scanned.pdf")
    with open(ROOT / "data" / "parts.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["equipment", "part_number", "description", "qty_per_unit"])
        writer.writerows(PARTS)
    library()
    print(f"samples written to {SAMPLES.relative_to(ROOT)}, library to {LIBRARY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
