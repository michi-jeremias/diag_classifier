import json

from sentence_transformers import SentenceTransformer, util
from diagnose_similarity.diagnose_similarity import DiagnoseSimilarity


# save diags as json
DIAGS = [
    "Abklärung instab. Ang. Pect.",
    "Bei klin. Hinweisen auf Zinkmangel z.B. Alopezie",
    "bei Neugeb. (Mutter HIV-pos.)",
    "benötigt strenge medizinische Indikation",
    "Biologica Therapie",
    "Blutentnahme aus der Vene - Kassenleistung",
    "CK erhöht",
    "Diabetes Mellitus",
    "einmalige Erstbestimmung",
    "exponierte Berufsgruppen (Nadelstichverletztung)",
    "Extrauteringravidität",
    "Gesamtbilirubin erhöht",
    "Harneiweiß negativ",
    "HAV IgM positiv",
    "HBc AK positiv",
    "HBc IgM wegen pathologischem Suchtest",
    "HBc IgM zur Verlaufskontrolle",
    "HBe AG wegen pathologischem Suchtest",
    "HBe AG zur Verlaufskontrolle",
    "HBe AK wegen pathologischem Suchtest",
    "HBe AK zur Verlaufskontrolle",
    "HBs AG positiv",
    "HBs AG positiv, vor Einstellung auf Biologica (inaktiv)",
    "HBV PCR positiv",
    "HCV AK positiv, zur Therapieentscheidung",
    "HCV PCR positiv",
    "HIV-infizierte zur Therapiekontrolle",
    "Hypercholesterinämie",
    "immer verrechenbar außer bei Drogenharn",
    "klinische Anamnese und Hauttestung erfolgt",
    "Lithiumpräparattherapie",
    "M-Gradienten in der Serumelektrophorese",
    "MUKIPA",
    "MUKIPA oder OP-Vorbereitung",
    "nur bei pathologischen Schilddrüsenhormonen oder bekannter Schilddrüsenerkrankung verrechenbar. (inaktiv)",
    "nur im Rahmen einer strukturierten Substitutionstherapie verrechenbar, gemeinsam mit KREA-U anfordern",
    "nur zur Anämiediagnostik",
    "nur zur Verlaufskontrolle von gesicherten malignen Tumoren (Magen oder Ovarial CA)",
    "nur zur Verlaufskontrolle von gesicherten malignen Tumoren (malignes Melanom)",
    "OP-Vorbereitung",
    "OP-Vorbereitung oder Transfusionsbedarf oder Antikoagulantientherapie",
    "pathologisches TSH",
    "PSA - Verlaufskontrolle wegen gesichertem malignen Tumor",
    "PSA - Vorsorge (> 50 Jahre)",
    "PSA wegen hereditärer Prädisposition (45 – 50 Jahre)",
    "RUB IgM positiv",
    "Schilddrüsen-Tumor",
    "SD-Therapie",
    "ß-HCG als Tumormarker",
    "strukturierte Substitutionstherapie",
    "Substitutionstherapie",
    "Therapiekontrolle",
    "Thrombophilieabklärung (Embolie, TVT)",
    "Toxo IgG positiv",
    "Transfusionsbedarf",
    "Überwachung von Risikoschwangerschaft",
    "bitte vermutete Autoimmunerkrankung in der Diagnose angeben",
    "V.a. Diabetes Mellitus",
    "V.a. Gerinnungsstörung",
    "V.a. Gewürzallergie, nicht-mögliche Hauttestung, klinische Anamnese erfolgt",
    "V.a. Gräserallergie, nicht-mögliche Hauttestung, klinische Anamnese erfolgt",
    "V.a. hämatologische Systemerkrankung",
    "V.a. Histaminintoleranz",
    "V.a. HIV Infektion",
    "V.a. Inhalationsallergie, nicht-mögliche Hauttestung, klinische Anamnese erfolgt",
    "V.a. Kontaktallergie, nicht-mögliche Hauttestung, klinische Anamnese erfolgt",
    "V.a. M.Reiter",
    "V.a. Morbus Basedow",
    "V.a. Morbus Bechterew",
    "V.a. Morbus Wilson",
    "V.a. Myocardinfarkt",
    "V.a. Nahrungsmittelallergie, nicht-mögliche Hauttestung, klinische Anamnese erfolgt",
    "V.a. Nussallergie, nicht-mögliche Hauttestung, klinische Anamnese erfolgt",
    "V.a. Pankreas-Insuffizienz",
    "V.a. schweren Immundefekt",
    "V.a. Serotoninmangel",
    "V.a. TBC",
    "V.a. Thrombophilie",
    "V.a. Thyreoiditis",
    "V.a. Verbrauchskoagulopathie",
    "V.a. Vit D3 Mangel",
    "V.a. Zöliakie",
    "Vor Einstellung auf Biologica",
    "Epilepsiebehandlung",
    "V. a. Muskelerkrangungen",
    "V. a. Epilepsie",
    "V. a. Hormonstörung",
    "Morbus Crohn/Colitis Diagnostik",
    "Abgrenzung CED - Reizdarm",
    "Verlaufskontrolle Colitis Ulcerosa",
    "Verlaufskontrolle Morbus Crohn",
    "ab 40 LJ bei hohem Risiko für Prostatakarzinom (erstgradige Verwandte, familiäre Häufung)",
    "ab 40 LJ bei bekannter o. V.a. BRCA1/2-Mutation",
    "bei Nachweis eines Hypogonadismus vor einer Testosteronsubstitution",
    "unter Testosteronsubstitution (im ersten Jahr halbjährlich, anschließend jährlich)",
    "abnormale digital-rektale Untersuchung bzw. konkreter Krebsverdacht (tastbare Knoten)",
    "Verlaufskontrolle von gesichertem malignen Tumor",
    "begründeter Verdacht auf Tumor",
    "Erhöhtes Gestoserisiko bei anamnestischer Präeklampsie ab der 20. SSW",
    "Eiweißausscheidung im Harn über 1g pro 24h bei Schwangerschaftshypertonie ab der 20 SSW",
    "Prostatahyperplasie",
    "Verrechnung an die AUVA",
    "Helicobacter Pylori Antigentest positiv",
    "patholog. Schilddrüsenhormone",
    "Schilddrüsenerkrankung",
    "Rheumatoide Arthritis (RA)=Chronische Polyarthritis(CP,PCP)",
    "Juvenile idiopathische Arthritis (JIA)= juvenile rheumatoide Arthritis (JRA)=juvenile chronische Arthritis (JCA), Morbus Still",
    "Systemischer Lupus erythematodes (LE,SLE)",
    "Sjögren Syndrom(SjS)",
    "Systemische Sklerodermie (SSc)",
    "Dermatomyositis (DM)",
    "Polymyositis (PM)",
    "Mixed Connective Tissue Disease=Sharp Syndrom=Mischkollagenose (MCTD)",
    "Autoimmunhepatitis (AIH)",
    "CREST-Syndrom (Calcinosis cutis, Raynaud-Phänomen, Esophagus-Motilitätsstörung, Sklerodaktylie, Teleangiektasie)",
    "Primär biliäre Cholangitis = Primär biliäre Cirrhose (PBC)",
    "Primär sklerosierende Cholangitis (PSC)",
    "Erhöhte Eiweißausscheidung",
    "V.a. Paraproteinämie",
    "Positiver Antigen-Test",
    "Negativer Antigentest, jedoch Symptomhäufung und stark ausgeprägte Intensität, sowie anamnestischer Kontakt zu einer COVID-19 erkrankten Person.",
    "Sachverhalt nicht eindeutig trotz Einsatz aller serologischen Mittel",
    "Sero-negativ trotz klinischer Erscheinungen, die auf HIV-Infektion schließen lassen",
    "Schwanger nach diagnostizierter HIV",
    "Akute HIV Infektion",
    "PAP III",
    "Abklärung unklarer Abstriche",
    "Patientinnen mit besonderem Risiko",
    "CoV-2 Infektionssymptome",
    "jährliches Screening",
    "Kontrolle nach pos. Ergebnis Voruntersuchung",
    "Im Rahmen der Myelomdiagnostik verrechenbar",
    "V.a. hämorrhagische Diathese",
    "Abklärung Gerinnungsstörung bei verlängerten Globaltests (PTZ-Quick und/oder PTT)",
    "Thrombophilie Diagnostik",
    "V.a. Sarkoidose/M. Böck",
]

diags_ids = {name: id + 1 for id, name in enumerate(DIAGS)}
with open("enml_diag.json", encoding="utf-8", mode="w") as f:
    json.dump(diags_ids, f, ensure_ascii=False, indent=4)


# sample texts
sentences = ["This is an example sentence", "Each sentence is converted"]
sentences = [
    "That is a happy person",
    "That is a happy  person",
    "That is a happy dog",
    "That is a very happy person",
    "Today is a sunny day",
]
sentences = [
    "Ich sitze im Büro",
    "Ich sitze im Buero",
    "Ich sitze im Büro",
    "Er sitzt im Büro",
    "Sie sitzt heute im Büro",
    "Ich esse einen Apfel",
]

# load model
# sentence-transformers/paraphrase-MiniLM-L6-v2
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
# sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# evaluate
embeddings = model.encode(sentences)

diag_embeddings = model.encode(DIAGS)
input_embedding = model.encode("Diabetes")
similarity = util.pytorch_cos_sim(input_embedding, diag_embeddings)


def get_input_embedding(input: str):
    input_embedding = model.encode(input)
    return input_embedding


def get_similarity(input_embedding, diag_embeddings):
    return util.pytorch_cos_sim(input_embedding, diag_embeddings)


embedding_0 = model.encode(sentences[0], convert_to_tensor=True)
embedding_1 = model.encode(sentences[1], convert_to_tensor=True)
embedding_2 = model.encode(sentences[2], convert_to_tensor=True)
embedding_3 = model.encode(sentences[3], convert_to_tensor=True)
util.pytorch_cos_sim(embedding_0, embedding_1)
util.pytorch_cos_sim(embedding_0, embedding_2)
util.pytorch_cos_sim(embedding_0, embedding_3)


model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

enml_diagsim = DiagnoseSimilarity(model)
enml_diagsim.show_similarity("Diabetes")
