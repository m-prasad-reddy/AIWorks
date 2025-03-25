import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Simulated `create_training_data` function (since it’s not provided)
def create_training_data():
    return [
        ("how many customers joined in 2023", "SELECT COUNT(*) FROM Customers WHERE JoinDate >= '2023-01-01' AND JoinDate < '2024-01-01'"),
        ("what is the total amount of orders", "SELECT SUM(TotalAmount) FROM Orders"),
        ("show me customer names", "SELECT FirstName, LastName FROM Customers"),
        ("list orders from 2023", "SELECT * FROM Orders WHERE OrderDate >= '2023-01-01' AND OrderDate < '2024-01-01'"),
        ("Show me all customer names and their join dates", "SELECT FirstName, LastName, JoinDate FROM Customers"),
        ("What is the average order amount in 2023?", "SELECT AVG(TotalAmount) FROM Orders WHERE OrderDate >= '2023-01-01' AND OrderDate < '2024-01-01'"),
        #("List all orders placed by Jane Smith", "SELECT * FROM Orders WHERE CustomerID = (SELECT CustomerID FROM Customers WHERE FirstName = 'Jane' AND LastName = 'Smith')"),
        ("List all orders placed by Jane Smith", "SELECT * FROM Orders o JOIN Customers c ON o.CustomerID = c.CustomerID WHERE c.FirstName = 'Jane' AND c.LastName = 'Smith'"),
        ("Get the email addresses of all customers", "SELECT Email FROM Customers"),
        ("How many orders were placed in June 2023?", "SELECT COUNT(*) FROM Orders WHERE OrderDate >= '2023-06-01' AND OrderDate < '2023-07-01'"),
    ]

# Step 1: Prepare the data
training_data = create_training_data()
questions = [q for q, _ in training_data]
sql_queries = [sql for _, sql in training_data]

# Step 2: Tokenization
question_tokenizer = Tokenizer()
question_tokenizer.fit_on_texts(questions)
question_sequences = question_tokenizer.texts_to_sequences(questions)
max_question_len = max(len(seq) for seq in question_sequences)
question_vocab_size = len(question_tokenizer.word_index) + 1

sql_tokenizer = Tokenizer()
sql_tokenizer.fit_on_texts(sql_queries)
sql_sequences = sql_tokenizer.texts_to_sequences(sql_queries)
max_sql_len = max(len(seq) for seq in sql_sequences)
sql_vocab_size = len(sql_tokenizer.word_index) + 1

# Pad sequences
encoder_input_data = pad_sequences(question_sequences, maxlen=max_question_len, padding='post')
decoder_input_data = pad_sequences(sql_sequences, maxlen=max_sql_len, padding='post')
decoder_target_data = np.zeros((len(sql_queries), max_sql_len, sql_vocab_size), dtype='float32')

# Prepare decoder target data (one-hot encoded)
for i, seq in enumerate(sql_sequences):
    for t, word_idx in enumerate(seq):
        if t > 0:  # Shift target by one timestep
            decoder_target_data[i, t-1, word_idx] = 1.0
    if len(seq) < max_sql_len:
        decoder_target_data[i, len(seq)-1:, 0] = 1.0  # Padding token

# Step 3: Define the model
encoder_inputs = Input(shape=(max_question_len,))
encoder_embedding = Embedding(input_dim=question_vocab_size, output_dim=64)(encoder_inputs)
encoder_lstm = LSTM(128, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(max_sql_len,))
decoder_embedding = Embedding(input_dim=sql_vocab_size, output_dim=64)(decoder_inputs)
decoder_lstm = LSTM(128, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(sql_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Step 4: Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=32, epochs=10)

# Step 5: Summary to confirm
model.summary()