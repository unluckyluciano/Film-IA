import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

#Carichiamo il dataset delle valutazioni e dei film
valutazioni = pd.read_csv('rating.csv')
valutazioni_porzione = valutazioni.sample(frac=0.001, random_state=42)  #riduzione della dimensione del dataset
film = pd.read_csv('movie.csv')

#Dividiamo il dataset di valutazioni in set di allenamento (80%) e set di test (20%)
dati_allenamento, dati_test = train_test_split(valutazioni_porzione, test_size=0.2, random_state=42)

#Creiamo la matrice utente-film usando solo il set di allenamento
matrice_utente_film = dati_allenamento.pivot_table(index='userId', columns='movieId', values='rating')

#Sostituiamo i valori mancanti con 0
matrice_utente_film.fillna(0, inplace=True)

similarita_utenti = cosine_similarity(matrice_utente_film)

#Creiamo un DataFrame per la similarità tra utenti
similarita_utenti_df = pd.DataFrame(similarita_utenti, index=matrice_utente_film.index, columns=matrice_utente_film.index)

valutazioni_previste = np.dot(similarita_utenti, matrice_utente_film)

#Creiamo un DataFrame per le valutazioni previste
valutazioni_previste_df = pd.DataFrame(valutazioni_previste, index=matrice_utente_film.index, columns=matrice_utente_film.columns)

def raccomanda_film(id_utente, numero_raccomandazioni=5):
    #Verifica se l'ID utente esiste nel DataFrame
    if id_utente not in valutazioni_previste_df.index:
        raise ValueError(f"L'utente con ID {id_utente} non esiste nel dataset di predizione.")

    #Ordiniamo i film in base alle valutazioni previste
    valutazioni_ordinate = valutazioni_previste_df.loc[id_utente].sort_values(ascending=False)

    #Film che l'utente non ha ancora visto
    film_visti = matrice_utente_film.loc[id_utente]
    film_raccomandati = valutazioni_ordinate[film_visti == 0].head(numero_raccomandazioni)

    #Aggiungiamo i titoli dei film
    film_raccomandati = pd.DataFrame(film_raccomandati).merge(film, left_index=True, right_on='movieId')

    return film_raccomandati[['movieId', 'title', 'genres']]

#Testiamo il sistema per un utente specifico
film_raccomandati = raccomanda_film(id_utente=18, numero_raccomandazioni=5)

#For per la stampa di titolo e genere singolarmente
for indice, riga in film_raccomandati.iterrows():
    print(f"Film: {riga['title']} (Genere: {riga['genres']})")


#Crea la matrice utente-film per il set di test
matrice_test_utente_film = dati_test.pivot_table(index='userId', columns='movieId', values='rating')

#Filtra per utenti e film che sono presenti nelle valutazioni previste dal modello
matrice_test_utente_film = matrice_test_utente_film[matrice_test_utente_film.index.isin(valutazioni_previste_df.index)]
matrice_test_utente_film = matrice_test_utente_film.loc[:, matrice_test_utente_film.columns.isin(valutazioni_previste_df.columns)]

#Previsioni del modello per il set di test
valutazioni_previste_test = valutazioni_previste_df.loc[matrice_test_utente_film.index, matrice_test_utente_film.columns]

mse = mean_squared_error(matrice_test_utente_film.fillna(0).values.flatten(), valutazioni_previste_test.fillna(0).values.flatten())
rmse = np.sqrt(mse)

print(f"Errore Quadratico Medio (MSE): {mse}")
print(f"Radice dell'Errore Quadratico Medio (RMSE): {rmse}")

film['genres'] = film['genres'].str.split('|')

#One-hot encoding dei generi per la ricerca dei film
encoder = OneHotEncoder()
generi_binari = encoder.fit_transform(film['genres'].apply(lambda x: ','.join(x)).values.reshape(-1, 1))

#Usiamo l'algoritmo K-NN per trovare i film più vicini in base ai generi
modello_knn = NearestNeighbors(n_neighbors=5, metric='cosine')
modello_knn.fit(generi_binari)

#Funzione per trovare i K film più simili a un dato film
def trova_film_simili(film_id, num_raccomandazioni=5):
    #Troviamo l'indice del film nel DataFrame
    indice_film = film.index[film['movieId'] == film_id].tolist()[0]

    #Trova i K film più vicini
    distanze, indici = modello_knn.kneighbors(generi_binari[indice_film], n_neighbors=num_raccomandazioni)

    film_simili = film.iloc[indici[0]]  #restituiamo i film più simili
    return film_simili[['movieId', 'title', 'genres']]

#Trova i 5 film più simili a un film con ID specifico
film_simili = trova_film_simili(film_id=1, num_raccomandazioni=20)
