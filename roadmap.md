# Roadmap: Clasificator de Companii pentru Taxonomia de Asigurări

## 1. Analiza și Explorarea Datelor (1-2 zile)

### 1.1 Explorarea Taxonomiei de Asigurări

- [ ] Vizualizarea și analiza structurii taxonomiei
- [ ] Identificarea categoriilor principale și subcategoriilor
- [ ] Analiza distribuției și specificității etichetelor
- [ ] Identificarea potențialelor relații ierarhice între etichete

### 1.2 Analiza Datelor despre Companii

- [ ] Verificarea calității și completitudinii datelor
- [ ] Analiza statistică a lungimii și complexității descrierilor
- [ ] Explorarea distribuției companiilor pe sectoare, categorii și nișe
- [ ] Identificarea corelațiilor dintre sectoarele existente și taxonomia de asigurări
- [ ] Evaluarea utilității fiecărui câmp (description, business_tags, sector, category, niche)

### 1.3 Documentarea Insights-urilor

- [ ] Crearea unui raport de analiză explorativă
- [ ] Formularea ipotezelor inițiale privind relația dintre companii și taxonomie

## 2. Preprocesarea Datelor (2-3 zile)

### 2.1 Preprocesarea Textului

- [ ] Curățarea textului (eliminarea caracterelor speciale, standardizare)
- [ ] Tokenizare și eliminarea cuvintelor irelevante (stopwords)
- [ ] Lematizare sau stemming pentru reducerea dimensionalității
- [ ] Extragerea entităților numite (NER) pentru identificarea termenilor relevanți

### 2.2 Feature Engineering

- [ ] Crearea bag-of-words sau TF-IDF pentru descrierile companiilor
- [ ] Generarea de embeddings pentru termenii relevanți
- [ ] Combinarea informațiilor din toate câmpurile disponibile
- [ ] Normalizarea și standardizarea feature-urilor numerice (dacă există)

### 2.3 Crearea Pipeline-ului de Preprocesare

- [ ] Implementarea unui pipeline automat de preprocesare
- [ ] Testarea și validarea pipeline-ului pe un subset de date
- [ ] Documentarea procesului și a deciziilor luate

## 3. Dezvoltarea Modelului de Clasificare (3-4 zile)

### 3.1 Implementarea Abordării bazate pe Similaritate

- [ ] Generarea reprezentărilor vectoriale pentru descrierile companiilor
- [ ] Generarea reprezentărilor vectoriale pentru etichetele din taxonomie
- [ ] Implementarea calculului de similaritate (cosinus, Jaccard, etc.)
- [ ] Dezvoltarea algoritmului de atribuire a etichetelor pe baza scorurilor de similaritate
- [ ] Evaluarea preliminară a rezultatelor

### 3.2 Implementarea Clasificatorului Supervizat (opțional)

- [ ] Crearea unui subset de date etichetat manual (100-200 exemple)
- [ ] Antrenarea unui clasificator multi-etichetă (OneVsRest cu SVM/Random Forest)
- [ ] Validarea cross-fold pentru optimizarea hiperparametrilor
- [ ] Evaluarea performanței pe setul de validare

### 3.3 Implementarea Abordării Zero-Shot (opțional)

- [ ] Configurarea unui model de limbaj pretrained (BERT, RoBERTa)
- [ ] Formularea problemei ca task de inferență textuală
- [ ] Implementarea predicției zero-shot pentru clasificare
- [ ] Evaluarea performanței abordării

### 3.4 Compararea și Selectarea Modelului

- [ ] Stabilirea metricilor de evaluare relevante
- [ ] Compararea performanței diferitelor abordări
- [ ] Selectarea celei mai promițătoare abordări sau crearea unui ensemble
- [ ] Documentarea procesului decizional și a rezultatelor

## 4. Evaluare și Optimizare (2-3 zile)

### 4.1 Validarea Manuală

- [ ] Evaluarea manuală a unui subset de predicții
- [ ] Identificarea pattern-urilor de eroare comune
- [ ] Documentarea cazurilor edge și a provocărilor

### 4.2 Optimizarea Modelului

- [ ] Ajustarea parametrilor și a pragurilor de decizie
- [ ] Implementarea strategiilor pentru gestionarea cazurilor dificile
- [ ] Rafinarea modelului pe baza feedback-ului din evaluarea manuală

### 4.3 Analiza Confuziilor

- [ ] Identificarea etichetelor frecvent confundate
- [ ] Analiza similitudinii semantice între etichetele problematice
- [ ] Dezvoltarea de reguli sau ajustări pentru a reduce confuziile

## 5. Scalabilitate și Optimizare pentru Producție (2 zile)

### 5.1 Analiza de Scalabilitate

- [ ] Testarea performanței pe volume mari de date
- [ ] Identificarea blocajelor și a punctelor de optimizare
- [ ] Implementarea strategiilor de paralelizare unde este necesar

### 5.2 Optimizarea Memoriei și Performanței

- [ ] Reducerea dimensionalității prin selecția feature-urilor
- [ ] Implementarea tehnicilor de compresie pentru modelele mari
- [ ] Optimizarea timpului de inferență pentru utilizare în timp real

### 5.3 Planul de Implementare în Producție

- [ ] Definirea fluxului de procesare pentru noi companii
- [ ] Stabilirea strategiei de actualizare a modelului
- [ ] Documentarea cerințelor de infrastructură și a dependențelor

## 6. Documentare și Prezentare (2 zile)

### 6.1 Documentarea Tehnică

- [ ] Descrierea arhitecturii soluției
- [ ] Documentarea detaliată a modelelor și a parametrilor
- [ ] Crearea unui ghid de utilizare și întreținere

### 6.2 Analiza Critică

- [ ] Evaluarea punctelor forte și a limitărilor soluției
- [ ] Discutarea asumărilor făcute și a impactului lor
- [ ] Propunerea direcțiilor de îmbunătățire viitoare

### 6.3 Pregătirea Prezentării

- [ ] Crearea unei prezentări concise și relevante
- [ ] Selecția exemplelor reprezentative pentru demonstrație
- [ ] Pregătirea pentru posibile întrebări și provocări

## 7. Livrarea Finală (1 zi)

### 7.1 Pregătirea Livrabilelor

- [ ] Generarea listei de companii clasificate (cu noua coloană "insurance_label")
- [ ] Finalizarea și organizarea codului sursă
- [ ] Compilarea documentației și a materialelor de prezentare

### 7.2 Verificarea Finală

- [ ] Verificarea completitudinii și corectitudinii livrabilelor
- [ ] Validarea finală a soluției pe un subset de date
- [ ] Asigurarea că toate cerințele din specificații sunt îndeplinite

## Timeline Estimat

- **Săptămâna 1**: Explorarea datelor, preprocesare și dezvoltarea inițială a modelului (Task-uri 1-3)
- **Săptămâna 2**: Evaluare, optimizare, scalabilitate și documentare (Task-uri 4-7)

## Resurse Necesare

- Biblioteci Python: pandas, numpy, scikit-learn, nltk/spaCy, transformers (pentru abordarea zero-shot)
- Acces la infrastructură de calcul pentru antrenarea modelelor mari (opțional)
- Resurse pentru validarea manuală a rezultatelor (timp și expertiză în domeniul asigurărilor)

## Potențiale Riscuri și Strategii de Atenuare

- **Risc**: Calitate scăzută a datelor de intrare
  - **Strategie**: Implementarea unor etape robuste de preprocesare și curățare
- **Risc**: Ambiguitate în taxonomia de asigurări
  - **Strategie**: Dezvoltarea unei măsuri de încredere în predicții și posibila adnotare manuală pentru cazurile ambigue
- **Risc**: Scalabilitate limitată pentru volumele mari de date
  - **Strategie**: Implementarea timpurie a testelor de scalabilitate și planificarea optimizărilor necesare
