"""Regenerate the test fixture PDF.

    python scripts/make_fixture.py

Nine encyclopedia entries across eighteen pages, each with a repeated running
head and a bare page number, so the cleaning rules have something real to
remove. Committed so tests and CI never need the 16 MB corpus.
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "mini_encyclopedia.pdf"

ENTRIES = {
    "IRON DEFICIENCY ANEMIA": [
        "Iron deficiency anemia is the most common nutritional deficiency worldwide. It develops when",
        "iron stores fall below the level required for normal erythropoiesis, most often through chronic",
        "blood loss, inadequate dietary intake, or impaired absorption in the duodenum. Presenting",
        "symptoms include fatigue, pallor, exertional dyspnoea and, in severe cases, pica. Diagnosis rests",
        "on a low serum ferritin with reduced transferrin saturation. Treatment is oral ferrous sulfate",
        "together with investigation and correction of the underlying source of loss.",
    ],
    "DIABETES MELLITUS TYPE 2": [
        "Type 2 diabetes mellitus is characterised by insulin resistance with relative insulin deficiency.",
        "Diagnosis is established by a fasting plasma glucose of 126 mg per deciliter or higher on two",
        "occasions, a two-hour value of 200 mg per deciliter on an oral glucose tolerance test, or a",
        "glycated hemoglobin of 6.5 percent or higher. Initial management combines dietary modification",
        "and physical activity with metformin as first-line pharmacotherapy. Long-term complications",
        "include retinopathy, nephropathy, peripheral neuropathy and accelerated atherosclerosis.",
    ],
    "APPENDICITIS": [
        "Appendicitis is inflammation of the vermiform appendix, usually following luminal obstruction",
        "by a fecalith or lymphoid hyperplasia. The classic presentation is periumbilical pain migrating",
        "to the right lower quadrant, accompanied by anorexia, nausea and low-grade fever. Rebound",
        "tenderness at McBurney point is characteristic on examination. Untreated, the appendix may",
        "perforate within twenty-four to seventy-two hours, producing localised abscess or generalised",
        "peritonitis. Definitive treatment is appendectomy, performed laparoscopically where possible.",
    ],
    "WOUND HEALING": [
        "Wound healing proceeds through four overlapping phases. Hemostasis begins within seconds as",
        "platelets aggregate and a fibrin clot forms. The inflammatory phase follows, with neutrophils",
        "and then macrophages clearing debris and bacteria over the first three days. Proliferation",
        "spans roughly day three to day twenty-one, during which fibroblasts deposit collagen, new",
        "capillaries form and epithelial cells migrate across the defect. Remodeling continues for up",
        "to a year as type III collagen is replaced by stronger type I collagen.",
    ],
    "HYPERTENSION": [
        "Hypertension is sustained elevation of arterial blood pressure. The kidney regulates pressure",
        "chiefly through the renin-angiotensin-aldosterone system: reduced renal perfusion triggers",
        "renin release, generating angiotensin II, a potent vasoconstrictor that also stimulates",
        "aldosterone and thereby sodium and water retention. Untreated hypertension leads to left",
        "ventricular hypertrophy, heart failure, stroke, chronic kidney disease and retinopathy.",
        "Management combines sodium restriction, weight reduction and antihypertensive therapy.",
    ],
    "IMMUNIZATION": [
        "Vaccines produce immunity by presenting the immune system with antigen in a form that does not",
        "cause disease. B lymphocytes generate antibody while memory B and T cells persist, allowing a",
        "faster and larger secondary response on later exposure. Live attenuated vaccines generally",
        "produce more durable immunity than inactivated preparations but are contraindicated in",
        "immunocompromised patients. Herd immunity protects those who cannot be vaccinated when a",
        "sufficient proportion of the population is immune.",
    ],
    "VITAMINS": [
        "Vitamins are organic compounds required in small amounts that the body cannot synthesise in",
        "sufficient quantity. The fat-soluble vitamins are A, D, E and K; they are stored in hepatic and",
        "adipose tissue and can accumulate to toxic concentrations. Water-soluble vitamins comprise the",
        "B-complex group and vitamin C, which are excreted in urine and require regular replacement.",
        "Deficiency states include night blindness with vitamin A, rickets and osteomalacia with",
        "vitamin D, and scurvy with prolonged vitamin C deficiency.",
    ],
    "STROKE": [
        "Stroke is the abrupt loss of neurological function caused by interruption of cerebral blood",
        "flow, either ischemic or hemorrhagic. Warning signs are sudden facial droop, unilateral arm",
        "weakness, slurred or confused speech, visual loss and sudden severe headache. Onset time",
        "determines eligibility for thrombolysis, which is why the time the patient was last seen well",
        "is recorded. Imaging distinguishes infarction from hemorrhage before any treatment is given.",
        "Secondary prevention addresses blood pressure, lipids, atrial fibrillation and smoking.",
    ],
    "ASTHMA": [
        "Asthma is a chronic inflammatory disorder of the airways producing reversible obstruction,",
        "bronchial hyperresponsiveness and variable airflow limitation. Long-term management is",
        "stepwise: inhaled corticosteroids form the controller foundation, with long-acting beta-2",
        "agonists added where control remains inadequate. Short-acting bronchodilators relieve acute",
        "symptoms but frequent use signals poor control. Trigger avoidance, inhaler technique review",
        "and a written action plan are as important as pharmacotherapy.",
    ],
}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=A4)
    page = 0

    for title, lines in ENTRIES.items():
        for part in range(2):
            c.setFont("Helvetica-Bold", 9)
            c.drawString(60, 800, title)              # running head
            c.setFont("Helvetica-Bold", 13)
            if part == 0:
                c.drawString(60, 770, title.title())
            c.setFont("Helvetica", 9.5)
            y = 745 if part == 0 else 770
            for line in lines:
                c.drawString(60, y, line)
                y -= 13
                c.drawString(60, y, line)
                y -= 13
            c.setFont("Helvetica", 8)
            c.drawString(300, 40, str(page + 1))      # bare page number
            c.showPage()
            page += 1

    c.save()
    print(f"{OUT}: {page} pages, {OUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
