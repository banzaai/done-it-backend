import truststore
truststore.inject_into_ssl()  # use Windows cert store - fixes enterprise SSL inspection

import os
import json
import dotenv

# Load .env before anything else
dotenv.load_dotenv()

import numpy as np
import requests
from utils import Utils
from typing import List, Dict, Any

# Chat model - must support HF Serverless Inference API (free tier, no gated access needed)
# Using Qwen2.5-7B-Instruct: Apache 2.0, no terms acceptance required
CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
CHAT_API = f"https://api-inference.huggingface.co/models/{CHAT_MODEL}/v1/chat/completions"

# Embedding model via HF Inference API - no local download needed
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_API = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL}"


def _embed(texts: List[str], token: str) -> np.ndarray:
    resp = requests.post(
        EMBEDDING_API,
        headers={"Authorization": f"Bearer {token}"},
        json={"inputs": texts, "options": {"wait_for_model": True}},
        timeout=30,
    )
    resp.raise_for_status()
    return np.array(resp.json(), dtype=np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T


class Startup:
    def __init__(self, meta_path="data/chunks.json"):
        os.makedirs("data", exist_ok=True)
        self.meta_path = meta_path
        self.token = Utils().token
        self.store: List[Dict[str, Any]] = []
        self._load_meta()

    def _load_meta(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
            if entries:
                texts = [e["text"] for e in entries]
                vecs = _embed(texts, self.token)
                for entry, vec in zip(entries, vecs):
                    self.store.append({**entry, "vec": vec})

    def _save_meta(self):
        data = [{"text": e["text"], "doc_id": e["doc_id"], "source": e["source"]}
                for e in self.store]
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be >= 0 and < chunk_size")
        chunks, start, n = [], 0, len(text)
        while start < n:
            end = min(start + chunk_size, n)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap
        return chunks

    def add_document_to_index(self, content: str, doc_id: str, source: str = "") -> int:
        chunks = self.chunk_text(content)
        if not chunks:
            return 0
        vecs = _embed(chunks, self.token)
        for chunk, vec in zip(chunks, vecs):
            self.store.append({"text": chunk, "doc_id": doc_id, "source": source, "vec": vec})
        self._save_meta()
        return len(chunks)

    def retrieve(self, query: str, k: int = 4) -> List[Dict]:
        if not self.store:
            return []
        q_vec = _embed([query], self.token)
        store_vecs = np.stack([e["vec"] for e in self.store])
        scores = _cosine_similarity(q_vec, store_vecs)[0]
        top_k = np.argsort(scores)[::-1][:k]
        return [{"score": float(scores[i]), **{kk: vv for kk, vv in self.store[i].items() if kk != "vec"}}
                for i in top_k]

    def query_index(self, query: str, k: int = 4) -> List[Dict]:
        return self.retrieve(query, k=k)

    def chat(self, messages: List[Dict]) -> str:
        """Call HF API if reachable, otherwise fall back to keyword-based response."""
        try:
            resp = requests.post(
                CHAT_API,
                headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
                json={"model": CHAT_MODEL, "messages": messages, "max_tokens": 512},
                timeout=10,
            )
            # Detect proxy block pages (ZScaler, etc.) returned as 200 HTML
            if resp.ok and resp.text.lstrip().startswith("<"):
                raise RuntimeError("API returned HTML (proxy block)")
            if not resp.ok:
                raise RuntimeError(f"HTTP {resp.status_code}")
            data = resp.json()
            if "error" in data:
                raise RuntimeError(f"Model error: {data['error']}")
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[chat] LLM API unavailable ({e}), using local fallback")
            return self._keyword_fallback(messages)

    def _keyword_fallback(self, messages: List[Dict]) -> str:
        """Keyword-based multilingual response about Done-it when LLM API is unreachable."""
        # Detect language from system prompt
        lang = "nl"
        for msg in messages:
            if msg["role"] == "system":
                c = msg["content"].lower()
                if "french" in c: lang = "fr"
                elif "english" in c: lang = "en"
                elif "spanish" in c: lang = "es"
                break

        user_msg = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_msg = msg["content"].lower()
                break

        # Per-language fallback responses
        fallbacks = {
            "nl": {
                "greet": "Hallo! Ik ben de virtuele assistent van Done-it. Ik help u graag met vragen over onze GPS-tijdregistratie en projectbeheer app. Hoe kan ik u helpen?",
                "price": "Done-it biedt flexibele abonnementsformules voor elk type bedrijf. Neem contact op via info@done-it.be of vraag een gratis demo aan op done-it.be zodat we u een voorstel op maat kunnen maken.",
                "gps": "Done-it registreert GPS-locaties en ritten automatisch en in realtime. Medewerkers kloppen in zodra ze hun wagen starten en de app legt automatisch de route, het begin- en eindadres en de afgelegde afstand vast.",
                "time": "Met Done-it registreren medewerkers hun werktijden automatisch via GPS of handmatig via de app. Overuren, pauzes en afwezigheden worden bijgehouden en zijn exporteerbaar naar populaire payroll-systemen.",
                "project": "Done-it koppelt uren en locaties automatisch aan projecten en werven. Leidinggevenden zien in realtime wie waar werkt en hoeveel uur er aan elk project gespendeerd wordt.",
                "app": "De Done-it app is beschikbaar voor zowel Android als iOS. Medewerkers klokken eenvoudig in en uit, voegen notities toe en bekijken hun planning, allemaal via hun smartphone.",
                "report": "Done-it genereert gedetailleerde rapporten over gewerkte uren, locaties en projecten. U kunt deze exporteren naar Excel of direct koppelen aan uw boekhoudsoftware.",
                "integration": "Done-it integreert naadloos met tools zoals Exact Online, TeamLeader en Yuki. Via onze API zijn ook maatwerk-integraties mogelijk met uw bestaand ERP-systeem.",
                "contact": "U kunt Done-it bereiken via info@done-it.be of via ons contactformulier op done-it.be/contact. Onze medewerkers helpen u graag verder!",
                "demo": "Vraag een gratis demo aan via done-it.be/aan-de-slag. Onze specialisten laten u live zien hoe Done-it uw administratie vereenvoudigt en uw team efficiënter maakt.",
                "fleet": "Done-it beheert uw volledig wagenpark: GPS-tracking van alle voertuigen, onderhoudsbeheer en brandstofverbruik. U heeft altijd controle over uw vloot.",
                "privacy": "Done-it neemt privacy serieus en voldoet volledig aan de GDPR-wetgeving. Alle gegevens worden veilig opgeslagen op Europese servers en medewerkers worden ingelicht over wat er geregistreerd wordt.",
                "generic": "Done-it is een Belgische app voor GPS-tijdregistratie, rittenregistratie en projectbeheer. Stel mij gerust een specifieke vraag over onze functies, prijzen of integraties — ik help u graag verder!"
            },
            "fr": {
                "greet": "Bonjour\u00a0! Je suis l'assistant virtuel de Done-it. Je vous aide volontiers avec vos questions sur notre application de pointage GPS et de gestion de projets. Comment puis-je vous aider\u00a0?",
                "price": "Done-it propose des formules d'abonnement flexibles pour tous les types d'entreprises. Contactez-nous via info@done-it.be ou demandez une d\u00e9mo gratuite sur done-it.be.",
                "gps": "Done-it enregistre automatiquement les positions GPS et les trajets en temps r\u00e9el. L'application suit automatiquement l'itin\u00e9raire, les adresses de d\u00e9part et d'arriv\u00e9e ainsi que la distance parcourue.",
                "time": "Avec Done-it, les employ\u00e9s enregistrent leurs heures de travail automatiquement via GPS ou manuellement via l'application. Heures suppl\u00e9mentaires, pauses et absences sont suivies et exportables.",
                "project": "Done-it relie automatiquement les heures et les localisations aux projets et chantiers. Les responsables voient en temps r\u00e9el qui travaille o\u00f9 et combien d'heures sont consacr\u00e9es \u00e0 chaque projet.",
                "app": "L'application Done-it est disponible pour Android et iOS. Les employ\u00e9s pointent facilement, ajoutent des notes et consultent leur planning depuis leur smartphone.",
                "report": "Done-it g\u00e9n\u00e8re des rapports d\u00e9taill\u00e9s sur les heures travaill\u00e9es, les localisations et les projets. Exportez-les vers Excel ou connectez-les directement \u00e0 votre logiciel comptable.",
                "integration": "Done-it s'int\u00e8gre parfaitement avec des outils comme Exact Online, TeamLeader et Yuki. Des int\u00e9grations sur mesure sont \u00e9galement possibles via notre API.",
                "contact": "Vous pouvez nous contacter via info@done-it.be ou via notre formulaire de contact sur done-it.be/contact. Notre \u00e9quipe se fera un plaisir de vous aider\u00a0!",
                "demo": "Demandez une d\u00e9mo gratuite sur done-it.be/aan-de-slag. Nos sp\u00e9cialistes vous montreront en direct comment Done-it simplifie votre administration.",
                "fleet": "Done-it g\u00e8re l'ensemble de votre flotte\u00a0: suivi GPS de tous les v\u00e9hicules, gestion de l'entretien et consommation de carburant. Vous gardez toujours le contr\u00f4le.",
                "privacy": "Done-it prend la vie priv\u00e9e au s\u00e9rieux et est enti\u00e8rement conforme au RGPD. Toutes les donn\u00e9es sont stock\u00e9es en toute s\u00e9curit\u00e9 sur des serveurs europ\u00e9ens.",
                "generic": "Done-it est une application belge de pointage GPS, d'enregistrement de trajets et de gestion de projets. N'h\u00e9sitez pas \u00e0 me poser une question sp\u00e9cifique\u00a0!"
            },
            "en": {
                "greet": "Hello! I'm the Done-it virtual assistant. I'm happy to help with your questions about our GPS time-tracking and project management app. How can I help you?",
                "price": "Done-it offers flexible subscription plans for every type of business. Contact us at info@done-it.be or request a free demo at done-it.be.",
                "gps": "Done-it automatically records GPS locations and trips in real time. The app tracks routes, start and end addresses, and distances driven automatically.",
                "time": "With Done-it, employees record their working hours automatically via GPS or manually through the app. Overtime, breaks, and absences are tracked and exportable to popular payroll systems.",
                "project": "Done-it automatically links hours and locations to projects and job sites. Managers see in real time who is working where and how many hours are spent on each project.",
                "app": "The Done-it app is available for both Android and iOS. Employees easily clock in and out, add notes, and view their schedule, all from their smartphone.",
                "report": "Done-it generates detailed reports on worked hours, locations, and projects. Export them to Excel or connect directly to your accounting software.",
                "integration": "Done-it integrates seamlessly with tools like Exact Online, TeamLeader, and Yuki. Custom integrations are also possible through our API.",
                "contact": "You can reach Done-it at info@done-it.be or through our contact form at done-it.be/contact. Our team is happy to help!",
                "demo": "Request a free demo at done-it.be/aan-de-slag. Our specialists will show you live how Done-it simplifies your administration and makes your team more efficient.",
                "fleet": "Done-it manages your entire fleet: GPS tracking of all vehicles, maintenance management, and fuel consumption. You always stay in control.",
                "privacy": "Done-it takes privacy seriously and is fully GDPR compliant. All data is securely stored on European servers and employees are informed about what is being recorded.",
                "generic": "Done-it is a Belgian app for GPS time-tracking, trip registration, and project management. Feel free to ask me a specific question about our features, pricing, or integrations!"
            },
            "es": {
                "greet": "\u00a1Hola! Soy el asistente virtual de Done-it. Estoy encantado de ayudarte con tus preguntas sobre nuestra app de registro de horas por GPS y gesti\u00f3n de proyectos. \u00bfC\u00f3mo puedo ayudarte?",
                "price": "Done-it ofrece planes de suscripci\u00f3n flexibles para todo tipo de empresas. Cont\u00e1ctanos en info@done-it.be o solicita una demo gratuita en done-it.be.",
                "gps": "Done-it registra autom\u00e1ticamente las ubicaciones GPS y los trayectos en tiempo real. La app rastrea rutas, direcciones de inicio y fin, y distancias recorridas.",
                "time": "Con Done-it, los empleados registran sus horas de trabajo autom\u00e1ticamente por GPS o manualmente a trav\u00e9s de la app. Horas extra, pausas y ausencias se rastrean y son exportables.",
                "project": "Done-it vincula autom\u00e1ticamente horas y ubicaciones a proyectos y obras. Los responsables ven en tiempo real qui\u00e9n trabaja d\u00f3nde y cu\u00e1ntas horas se dedican a cada proyecto.",
                "app": "La app Done-it est\u00e1 disponible para Android e iOS. Los empleados fichan f\u00e1cilmente, a\u00f1aden notas y consultan su planificaci\u00f3n desde su smartphone.",
                "report": "Done-it genera informes detallados sobre horas trabajadas, ubicaciones y proyectos. Exp\u00f3rtalos a Excel o con\u00e9ctalos directamente a tu software contable.",
                "integration": "Done-it se integra perfectamente con herramientas como Exact Online, TeamLeader y Yuki. Tambi\u00e9n son posibles integraciones personalizadas a trav\u00e9s de nuestra API.",
                "contact": "Puedes contactar con Done-it en info@done-it.be o a trav\u00e9s de nuestro formulario de contacto en done-it.be/contact. \u00a1Nuestro equipo estar\u00e1 encantado de ayudarte!",
                "demo": "Solicita una demo gratuita en done-it.be/aan-de-slag. Nuestros especialistas te mostrar\u00e1n en directo c\u00f3mo Done-it simplifica tu administraci\u00f3n.",
                "fleet": "Done-it gestiona toda tu flota: seguimiento GPS de todos los veh\u00edculos, gesti\u00f3n de mantenimiento y consumo de combustible. Siempre tienes el control.",
                "privacy": "Done-it se toma la privacidad en serio y cumple totalmente con el RGPD. Todos los datos se almacenan de forma segura en servidores europeos.",
                "generic": "Done-it es una app belga de registro de horas por GPS, registro de trayectos y gesti\u00f3n de proyectos. \u00a1No dudes en hacerme una pregunta espec\u00edfica!"
            }
        }

        fb = fallbacks.get(lang, fallbacks["nl"])
        kw = user_msg

        if any(w in kw for w in ["hallo", "hello", "hi ", "goeiedag", " dag", "wie ben", "wie zijn", "bonjour", "salut", "hola", "buenos"]):
            return fb["greet"]
        if any(w in kw for w in ["prijs", "kost", "pricing", "price", "kosten", "abonnement", "formule", "tarief", "prix", "precio", "tarif"]):
            return fb["price"]
        if any(w in kw for w in ["gps", "traceer", "tracking", "locatie", "localisatie", "rittenregist", "rit", "trajet", "localisation", "ubicaci"]):
            return fb["gps"]
        if any(w in kw for w in ["time", "uurregistratie", "tijdregistratie", "klokken", "werktijden", "uren", "aanwezigheid", "controle", "pointage", "horaire", "horas", "fichar"]):
            return fb["time"]
        if any(w in kw for w in ["project", "werf", "opdracht", "taak", "planning", "agenda", "chantier", "obra", "tarea"]):
            return fb["project"]
        if any(w in kw for w in ["app", "android", "iphone", "ios", "mobiel", "smartphone", "telefoon", "t\u00e9l\u00e9phone", "m\u00f3vil"]):
            return fb["app"]
        if any(w in kw for w in ["rapportering", "rapport", "export", "excel", "overzicht", "statistieken", "analyse", "report", "informe"]):
            return fb["report"]
        if any(w in kw for w in ["integratie", "koppeling", "exact", "teamleader", "yuki", "boekhoud", "payroll", "loon", "api", "erp", "int\u00e9gration", "integraci"]):
            return fb["integration"]
        if any(w in kw for w in ["contact", "bellen", "email", "bereiken", "info@", "adres", "kantoor", "bureau", "oficina"]):
            return fb["contact"]
        if any(w in kw for w in ["demo", "probeer", "trial", "gratis", "testen", "starten", "beginnen", "essayer", "gratuit", "probar"]):
            return fb["demo"]
        if any(w in kw for w in ["wagenparkbeheer", "voertuig", "vloot", "auto", "bestelwagen", "truck", "onderhoud", "flotte", "v\u00e9hicule", "flota", "veh\u00edculo"]):
            return fb["fleet"]
        if any(w in kw for w in ["privacy", "gdpr", "avg", "veiligheid", "gegevens", "data", "rgpd", "confidentialit", "privacidad"]):
            return fb["privacy"]
        return fb["generic"]

    async def answer_with_rag(self, question: str, k: int = 4) -> str:
        hits = self.retrieve(question, k=k)
        context = "\n\n".join(
            [f"[doc:{h['doc_id']}] {h['text']}" for h in hits]
        ) or "No relevant context found."
        msgs = [
            {"role": "system", "content": "Answer using the provided context. If insufficient, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
        ]
        return self.chat(msgs)
