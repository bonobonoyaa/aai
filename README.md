# wdm
# [EX3 Implementation of GSP Algorithm In Python](url)
```py
def generate_candidates(dataset, k):
    candidates = defaultdict(int)
    for sequence in dataset:
        for itemset in combinations(sequence, k):
            candidates[itemset] += 1
    return {item: support for item, support in candidates.items() if support >= min_support}

def gsp(dataset, min_support):
    frequent_patterns = defaultdict(int)
    k = 1
    while True:
        candidates = generate_candidates(dataset, k)
        # Prune candidates with support less than min_support
        if not candidates:
            break
        frequent_patterns.update(candidates)
        k += 1
    return frequent_patterns
```
# [EX4 Implementation of Cluster and Visitor Segmentation for Navigation patterns](url)
```py
import pandas as pd
visitor_df = pd.read_csv('/content/clustervisitor.csv')

age_groups = {
    'Young': visitor_df['Age'] <= 30,
    'Middle-aged': (visitor_df['Age'] > 30) & (visitor_df['Age'] <= 50),
    'Elderly': visitor_df['Age'] > 50
}

for group, condition in age_groups.items():
    visitors_in_group = visitor_df[condition]
    print(f"Visitors in {group} age group:")
    print(visitors_in_group)

#Visualization:
import matplotlib.pyplot as plt
visitor_counts=[]

for group,condition in age_groups.items():
  visitors_in_group=visitor_df[condition]
  visitor_counts.append(len(visitors_in_group))

age_group_labels=list(age_groups.keys())
```
# [EX5 Information Retrieval Using Boolean Model in Python](url)
```py
  def create_documents_matrix(self, documents):
          terms = list(self.index.keys())
          num_docs = len(documents)
          num_terms = len(terms)
  
          self.documents_matrix = np.zeros((num_docs, num_terms), dtype=int)
  
          for i, (doc_id, text) in enumerate(documents.items()):
              doc_terms = text.lower().split()
              for term in doc_terms:
                  if term in self.index:
                      term_id = terms.index(term)
                      self.documents_matrix[i, term_id] = 1

  def print_all_terms(self):
    print("\nAll terms in Documents:")
    terms_list = list(self.index.keys())
    terms_list.sort()
    print(terms_list)

  def print_documents_matrix_table(self):
    print("\nTerm Documents Matrix :")
    df = pd.DataFrame(self.documents_matrix, columns=self.index.keys())
    print(df)

  def boolean_search(self, query):
      query = query.lower()
      query_terms = query.split()
      results = None

      for term in query_terms:
        if term in self.index:
          if results is None:
            results = self.index[term]
          else:
            if query[0] == 'and':
              results = results.intersection(self.index[term])
            elif query[0] == 'or':
              results = results.union(self.index[term])
            elif query[0] == 'not':
              results = results.difference(self.index[term])
      return results if results else set()
```
# [EX6 Information Retrieval Using Vector Space Model in Python](url)
```py
def search(query, tfidf_matrix, tfidf_vectorizer):
    preprocessed_query = preprocess_text(query)
    query_vector = tfidf_vectorizer.transform([preprocessed_query])

    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)

    sorted_indexes = similarity_scores.argsort()[0][::-1]

    results = [(documents[i], similarity_scores[0, i]) for i in sorted_indexes]
    return results
```
# [EX7 Implementation of Link Analysis using HITS Algorithm](url)
```py
def hits_algorithm(adjacency_matrix, max_iterations=100, tol=1.0e-6):
    num_nodes = len(adjacency_matrix)
    authority_scores = np.ones(num_nodes)
    hub_scores = np.ones(num_nodes)
    
    for i in range(max_iterations):
        new_authority_scores = np.dot(adjacency_matrix.T, hub_scores)
        new_authority_scores /= np.linalg.norm(new_authority_scores, ord=2)

        new_hub_scores = np.dot(adjacency_matrix, new_authority_scores)
        new_hub_scores /= np.linalg.norm(new_hub_scores, ord=2) 
        
        authority_diff = np.linalg.norm(new_authority_scores - authority_scores, ord=2)
        hub_diff = np.linalg.norm(new_hub_scores - hub_scores, ord=2)
        
        if authority_diff < tol and hub_diff < tol:
            break
        
        authority_scores = new_authority_scores
        hub_scores = new_hub_scores
    
    return authority_scores, hub_scores
```
# [EX8 Web Scraping On E-commerce platform using BeautifulSoup](url)
```py
def get_snapdeal_products(search_query):
    url = f'https://www.snapdeal.com/search?keyword={search_query.replace(" ", "%20")}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'
    }

    response = requests.get(url, headers=headers)
    products_data = []

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        products = soup.find_all('div', {'class': 'product-tuple-listing'})

        for product in products:
            title = product.find('p', {'class': 'product-title'})
            price = product.find('span', {'class': 'product-price'})
            if price:
                product_price = convert_price_to_float(price.get('data-price', '0'))
            else:
                product_price = 0.0  # Default to 0 if no price found
            rating = product.find('div', {'class': 'filled-stars'})  # Assuming rating is shown with this class

            if title and price:
                product_name = title.text.strip()
                product_rating = rating['style'].split(';')[0].split(':')[-1] if rating else "No rating"
                products_data.append({
                    'Product': product_name,
                    'Price': float(product_price),
                    'Rating': product_rating
                })
                print(f'Product: {product_name}')
                print(f'Price: {product_price}')
                print(f'Rating: {product_rating}')
                print('---')

    else:
        print('Failed to retrieve content')

    return products_data
```
