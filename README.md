#這是一個能用中文自然語言控制和操作Linux的AI助理，使用者能夠使用"聊天"的方式來操控Linux系統，系統會根據語意並執行對應操作。

#使用方法:
執行assistant.py，系統會提問，只需要將你的需求輸入，系統會判斷你的語意並分成四類，分別是操作，教學，聊天，模糊。

操作:系統會直接生成對應指令。
教學:使用者可能有疑問都會進入到這個模式，系統會提供教學。
聊天:系統判斷聊天話題可能跟Linux無關，就會進入聊天模式。
模糊:使用者輸入的語句可能不夠清晰，系統無法判斷得很清楚就會分為此類，此模式它會再次詢問你的需求。

#檔案:
assistant.py 主程式(使用介面)
train.py 訓練分類器模型程式(啟動後會生成或覆蓋classifier.pkl和vectorizer.pkl兩個程式)
data.csv 訓練資料(格式為text,lable)
classifier.pkl 訓練好的模型
vectorizer.pkl 對應的向量器

#使用須知:

1.你必須要先有一個自己的OpenAI API，請將你的API填入assistant.py的程式中
2.建立使用環境:請先輸入這行指令 pip install openai pandas scikit-learn joblib jieba
3.你可以在data.csv這個檔案中填入資料使分類器更精準
4.填入好記得執行train.py


