
from math import log

class NaiveBayesSpamFilter:

    def __init__(self):
        # Estado aprendido (se llena en entrenar)
        self.vocabulario = None

        self.log_prob_spam = None
        self.log_prob_no_spam = None

        self.log_prior_spam = None
        self.log_prior_no_spam = None

        self.log_unk_spam = None
        self.log_unk_no_spam = None

    def _tokenizar(self, texto: str):
      texto = texto.lower()
      palabras = texto.split()
      return palabras

    def _contar_palabras_por_clase(self, emails):
      conteos_spam = {}
      conteos_no_spam = {}
      total_palabras_spam = 0
      total_palabras_no_spam = 0
      n_spam = 0
      n_no_spam = 0

      for texto, es_spam in emails:
          palabras = self._tokenizar(texto)

          if es_spam:
              n_spam += 1
              for palabra in palabras:
                  conteos_spam[palabra] = conteos_spam.get(palabra, 0) + 1
                  total_palabras_spam += 1

          else:
              n_no_spam += 1
              for palabra in palabras:
                  conteos_no_spam[palabra] = conteos_no_spam.get(palabra, 0) + 1
                  total_palabras_no_spam += 1

      return conteos_spam, conteos_no_spam, total_palabras_spam, total_palabras_no_spam, n_spam, n_no_spam

    def _calcular_probabilidades(self, conteos, total_palabras, vocabulario):
      probabilidades = {}

      for palabra in vocabulario:
          conteo = conteos.get(palabra, 0)
          P = (conteo + 1) / (total_palabras + len(vocabulario))

          probabilidades[palabra] = log(P)

      return probabilidades

    def _prob_mensaje_dado_clase(self, mensaje, prob_palabras_clase, prior_clase, log_unk):
      palabras = self._tokenizar(mensaje)
      prob = prior_clase

      for palabra in palabras:

          p_palabra = prob_palabras_clase.get(palabra, log_unk)
          prob += p_palabra

      return prob

    def score(self, mensaje):
      if self.log_prob_spam is None:
          raise ValueError("Modelo no entrenado")

      s_spam = self._prob_mensaje_dado_clase(
          mensaje,
          self.log_prob_spam,
          self.log_prior_spam,
          self.log_unk_spam,
      )

      s_no = self._prob_mensaje_dado_clase(
          mensaje,
          self.log_prob_no_spam,
          self.log_prior_no_spam,
          self.log_unk_no_spam,
      )

      return s_spam, s_no


    def entrenar(self, emails):

        if len(emails) == 0:
            raise ValueError("La lista de emails está vacía")

        conteos_spam, conteos_no_spam, total_spam, total_no_spam, n_spam, n_no_spam = self._contar_palabras_por_clase(emails)

        if n_spam == 0 or n_no_spam == 0:
            raise ValueError("La lista de spam o no spam está vacía")

        self.vocabulario = set(conteos_spam) | set(conteos_no_spam)
        total_emails = len(emails)

        self.log_prior_spam = log(n_spam / total_emails)
        self.log_prior_no_spam = log(n_no_spam / total_emails)

        self.log_prob_spam = self._calcular_probabilidades(conteos_spam, total_spam, self.vocabulario)
        self.log_prob_no_spam = self._calcular_probabilidades(conteos_no_spam, total_no_spam, self.vocabulario)

        V = len(self.vocabulario)

        self.log_unk_spam = log(1 / (total_spam + V))
        self.log_unk_no_spam = log(1 / (total_no_spam + V))

        return None

    def predecir(self, mensaje, threshold=0.0):
        s_spam, s_no = self.score(mensaje)
        delta = s_spam - s_no

        if delta >= threshold:
            return "spam", s_spam, s_no, delta
        else:
            return "no_spam", s_spam, s_no, delta

    def evaluar(self, emails):

        pred_label = ""
        aciertos = 0

        for texto, es_spam in emails:
          pred_label = self.predecir(texto)[0]
          pred_is_spam = (pred_label == "spam")

          if pred_is_spam == es_spam:
            aciertos += 1

        return aciertos / len(emails)


    def matriz_confusion(self, emails):
        pred_label = ""
        TP = FP = TN = FN = 0

        for texto, es_spam in emails:
          pred_label = self.predecir(texto)[0]
          pred_is_spam = (pred_label == "spam")
          real = es_spam

          if pred_is_spam and real: TP += 1

          if pred_is_spam and not real: FP += 1

          if not pred_is_spam and not real: TN += 1

          if not pred_is_spam and real: FN += 1

        return TP, FP, TN, FN

    def precision(self, emails):
        tp, fp, tn, fn = self.matriz_confusion(emails)
        denom = tp + fp
        if denom == 0:
            return 0.0
        return tp / denom


    def recall(self, emails):
        tp, fp, tn, fn = self.matriz_confusion(emails)
        denom = tp + fn
        if denom == 0:
            return 0.0
        return tp / denom

emails = [
    ("gana dinero ahora", True),
    ("dinero gratis ya", True),
    ("reunion ahora", False),
    ("nos vemos ya", False),
]
f = NaiveBayesSpamFilter()
f.entrenar(emails)

#print(f.predecir("ahora dinero"))
#print(f.predecir("reunion ahora"))
#print(f.predecir("palabra_que_no_existe"))

print(f.evaluar(emails))
print(f.matriz_confusion(emails))

#print(f.predecir("dinero gratis", threshold=0.0))
#print(f.predecir("dinero gratis", threshold=1.0))
#print(f.predecir("dinero gratis", threshold=5.0))


print("/n Comparacion con skalearn:")

emails = [
    ("gana dinero ahora", True),
    ("dinero gratis ya", True),
    ("reunion ahora", False),
    ("nos vemos ya", False),
]

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform([t for t, _ in emails])
y = [int(label) for _, label in emails]

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB(alpha=1.0)  # Laplace
model.fit(X, y)

pred = model.predict(X)

from sklearn.metrics import accuracy_score, confusion_matrix

print("Accuracy:", accuracy_score(y, pred))
print("Confusion matrix:")
print(confusion_matrix(y, pred))
