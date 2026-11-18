from fastapi import FastAPI 
from pydantic import BaseModel
import joblib

app= FastAPI()
model = joblib.load("model.pkl")

class flowerdata(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    
@app.get("/bio")
def bio():
    return {
        'name': 'my name is shivam',
        'age': 20,
        'city': 'delhi'
    }

    
@app.get("/")    
def home():
    return {"message": "Model is running"}

@app.post("/predict")
def predict(data: flowerdata):
    input_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    
    prediction = model.predict(input_data)
    flower_names = ["setosa", "versicolor", "virginica"]
    result = flower_names[int(prediction[0])]
    return {"prediction": result}


class data(BaseModel):
    name:str
    age: int
    city: str
    course: str
    
@app.post("/data")
def data(data:data):
    return{data.name,
           data.age,
           data.city,
           data.course}    