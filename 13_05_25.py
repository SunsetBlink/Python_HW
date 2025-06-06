import pandas as pd
import numpy as np

data = pd.read_csv("spam.csv")
data["Spam"] = data["Category"].apply(lambda x: 1 if x=="spam" else 0)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
X = vect.fit_transform(data["Message"])
w = vect.get_feature_names_out()
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
#model = Pipeline([("vect",CountVectorizer()),("NB",MultinomialNB())])
#model = Pipeline([("vect",CountVectorizer()),("NB",GaussianNB())])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data["Message"],data["Spam"],test_size=0.3)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_predict, y_test))

#second file
data = pd.read_csv("phishing.csv")
X = data.drop(columns="class")
Y = pd.DataFrame(data["class"])
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
model = dt.fit(X_train,y_train)
dt_predict = model.predict(X_test)
print(accuracy_score(X_test,y_test))

from os import environ

from bs4 import BeautifulSoup as bs

soup = bs(html_content, "lxml")

title = soup.find("title")
print(title)
print(type(title))
print(title.text)

# print(soup.body.text)
print(soup.body.p)

pList = soup.body.find_all("p")
for p in enumerate(pList):
    print(p.text)
    print("---------------")

print([bullet.text for bullet in soup.body.find_all("li")])

p2 = soup.find(id="paragraph 2")
print(p2.text)

divAll = soup.find_all("div")
print(divAll)
divClassText =  soup.find_all("div", class_ = "text")
print(divClassText)
for div in divClassText:
    id = div.ger("id")
soup.body.find(id="paragraph0").decompose()
soup.body.find(id="paragraph1")

new_p = soup.new_tag("p")
new_p.string = "new"
soup.find(id = "empty").append(new_p)

from urlib.request import urlopen

url = "https://yandex.ru"
html_content = urlopen(url).read(
)
sp = bs(html_content,"lxml")


