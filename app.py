from flask import Flask, request,render_template
from flask_restful import Resource, Api
import httpx
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertModel
import faiss
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
api = Api(app)
CORS(app)

# Define functions
def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embedding

def store_embedding_in_faiss(index, key, embedding):
    embedding = embedding.astype('float32')
    embedding = np.reshape(embedding, (1, -1))
    index.add(embedding)

def fetch_course_data(url):
    response = httpx.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        course_data = []
        for course_elem in soup.find_all('div', class_='single-courses-box'):
            title_elem = course_elem.find('h3').find('a')
            title = title_elem.text.strip() if title_elem else None

            description_elem = course_elem.find('p')
            description = description_elem.text.strip() if description_elem else None

            img_elem = course_elem.find('img')
            image_url = img_elem['src'].strip() if img_elem and 'src' in img_elem.attrs else None

            price_elem = course_elem.find('span', class_='price-per-session')
            price_per_session = price_elem.text.strip() if price_elem else None

            lessons_elem = course_elem.find('li', class_='flaticon-agenda')
            lessons = lessons_elem.text.strip() if lessons_elem else None

            view_details_link_elem = course_elem.find('a', class_='BookDemo-btn')
            view_details_link = f"https://brainlox.com/"+view_details_link_elem['href'].strip() if view_details_link_elem and 'href' in view_details_link_elem.attrs else None
            
            
            course_info = {
                "title": title,
                "description": description,
                "image_url": image_url,
                "price_per_session": price_per_session,
                "lessons": lessons,
                "view_details_link": view_details_link
            }
            course_data.append(course_info)

        return course_data
    else:
        return []
    
    
class CourseData(Resource):
    def get(self):
        url = "https://brainlox.com/courses/category/technical"
        course_data = fetch_course_data(url)
        return course_data

class Recommendations(Resource):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

       
        dimension = 768 
        self.index = faiss.IndexFlatIP(dimension) 
        url = "https://brainlox.com/courses/category/technical"
        self.course_data = fetch_course_data(url)
        self.populate_faiss_index()  
    def get(self):
        query = request.args.get('query', '')
        recommendations = self.get_recommendations(query)
        if recommendations:
            response_data = {
                'response': f"Sure! I can help you find courses. Based on your query, you might be interested in '{recommendations[0].get('title', 'No title')}' and similar courses. \n\nHere are the details:\n\n" +
                f"Description: {recommendations[0].get('description', 'No description available')}\n" +
                f"Price per Session: {recommendations[0].get('price_per_session', 'No price mentioned ')}\n"+
                f"View Details: <a href='{recommendations[0].get('view_details_link')}' target='_blank'>Click here</a>",
                'additional_data': 'Any additional data here',
            }
            return response_data
        else:
            return {'response': 'No recommendations',}
    
    def post(self):
        data = request.get_json()
        query = data.get('query', '')
        recommendations = self.get_recommendations(query)
        if recommendations:
            response_data = {
                'response': f"Sure! I can help you find courses. Based on your query, you might be interested in '{recommendations[0].get('title', 'No title')}' and similar courses. \n\nHere are the details:\n\n" +
                f"Description: {recommendations[0].get('description', 'No description available')}\n" +
                f"Price per Session: {recommendations[0].get('price_per_session', 'No price mentioned ')}\n" +
                f"View Details: <a href='{recommendations[0].get('view_details_link')}' target='_blank'>Click here</a>",
                'additional_data': 'Any additional data here',
            }
            return response_data
        else:
            return {'response': 'No recommendations', 'additional_data': None}
    

    def populate_faiss_index(self):
        for course_info in self.course_data:
            title_embedding = get_embedding(course_info['title'], self.tokenizer, self.model)
            store_embedding_in_faiss(self.index, course_info['title'], title_embedding)

    def get_recommendations(self, query, top_k=5):
        query_embedding = get_embedding(query, self.tokenizer, self.model)
        query_embedding = query_embedding.astype('float32')
        query_embedding = np.reshape(query_embedding, (1, -1))

        _, indices = self.index.search(query_embedding, top_k)
        recommendations = [self.course_data[i] for i in indices[0]]
        return recommendations

# Add resources to the API
api.add_resource(CourseData, '/course-data')
api.add_resource(Recommendations, '/recommendations')


def get_chatbot_response(query, recommendations_resource):


    # Existing logic ...
    if 'course' in query.lower():

        recommendations = recommendations_resource.get_recommendations(query)
        
        if recommendations:
          
            recommended_course_title = recommendations[0].get('title', 'a course')

          
            return f"Sure! I can help you find courses. Based on your query, you might be interested in '{recommended_course_title}' and similar courses."
        else:
           
            return "Sure! I can help you find courses. Just ask me about a specific topic."
   
    return "I'm sorry, I didn't understand that. Feel free to ask me about courses."



chatbot_recommendations = Recommendations()

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    user_query = data.get('query', '')
    print("User Query:", user_query)  
    chatbot_response = get_chatbot_response(user_query, chatbot_recommendations)
    return {'response': chatbot_response}

if __name__ == '__main__':
    app.run(debug=True)
