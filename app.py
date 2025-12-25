import time
from flask import Flask, render_template, request, jsonify
from pathlib import Path

# ====================  EfficientSemanticSearcher 类（已增强 FAISS 支持） ====================
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle
import os

class EfficientSemanticSearcher:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2',
                 vocab_path='vocab_clean.csv', cache_dir='./cache'):   #您可以改这里的csv路径
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
       
        self.embeddings_file = self.cache_dir / 'embeddings.npy'
        self.metadata_file = self.cache_dir / 'metadata.pkl'
        self.faiss_index_file = self.cache_dir / 'faiss_index.bin'  
       
        print("正在加载模型...")
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
       
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"词汇表文件 {vocab_path} 不存在")
       
        self.df = pd.read_csv(vocab_path)
        self.words = self.df['word'].tolist()
        self.meanings = self.df['meaning'].tolist()
       
        print(f"加载了 {len(self.words):,} 个单词")
       
        self.load_or_compute_embeddings()
        self.build_or_load_faiss_index()  
       
    def load_or_compute_embeddings(self):
        if self.embeddings_file.exists() and self.metadata_file.exists():
            print("正在从缓存加载嵌入...")
            with open(self.metadata_file, 'rb') as f:
                metadata = pickle.load(f)
           
            if (len(self.words) == metadata['num_words'] and
                metadata['vocab_hash'] == hash(tuple(self.words[:1000]))):
                self.word_embeddings = np.load(self.embeddings_file)
                print(f"成功加载缓存嵌入 ({self.word_embeddings.shape[0]:,} 个向量)")
                return
       
        print("未找到有效缓存，正在计算嵌入（首次运行较慢，以后秒开）...")
        start_time = time.time()
       
        batch_size = 1000
        all_embeddings = []
       
        for i in range(0, len(self.words), batch_size):
            batch = self.words[i:i+batch_size]
            embs = self.model.encode(
                batch,
                convert_to_tensor=False,
                show_progress_bar=True,
                batch_size=64,
                normalize_embeddings=True
            )
            all_embeddings.append(embs)
       
        self.word_embeddings = np.vstack(all_embeddings).astype(np.float16)
       
        np.save(self.embeddings_file, self.word_embeddings)
       
        metadata = {
            'num_words': len(self.words),
            'vocab_hash': hash(tuple(self.words[:1000])),
            'timestamp': time.time()
        }
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
       
        print(f"嵌入计算完成，用时 {time.time() - start_time:.1f} 秒，已保存缓存")

    def build_or_load_faiss_index(self):
        try:
            import faiss
            self.faiss_available = True
        except ImportError:
            print("FAISS 未安装，使用 numpy 矩阵搜索（仍很快）")
            self.faiss_available = False
            return
        
        if self.faiss_index_file.exists():
            print("正在加载 FAISS 索引...")
            self.faiss_index = faiss.read_index(str(self.faiss_index_file))
            print(f"FAISS 索引加载完成（{self.faiss_index.ntotal:,} 个向量）")
            return
       
        print("正在构建 FAISS 索引（首次较慢，以后秒开）...")
        start_time = time.time()
       
        dimension = self.word_embeddings.shape[1]
        # 使用 IndexFlatIP（精确内积搜索，因为已归一化 = 余弦相似度）
        self.faiss_index = faiss.IndexFlatIP(dimension)
        
        # 添加向量（需转为 float32）
        vectors_float32 = self.word_embeddings.astype(np.float32)
        self.faiss_index.add(vectors_float32)
        
        # 保存索引
        faiss.write_index(self.faiss_index, str(self.faiss_index_file))
        
        print(f"FAISS 索引构建完成，用时 {time.time() - start_time:.1f} 秒，已保存")

    def search(self, query, top_k=10, threshold=0.3):
        if self.faiss_available:
            return self.search_with_faiss(query, top_k=top_k, threshold=threshold)
        else:
            return self.search_with_numpy(query, top_k=top_k, threshold=threshold)

    def search_with_faiss(self, query, top_k=10, threshold=0.3):
        import faiss
        query_emb = self.model.encode(
            query,
            convert_to_tensor=False,
            normalize_embeddings=True
        ).astype(np.float32).reshape(1, -1)
       
        # 多搜一些以防阈值过滤
        k = min(top_k * 5, len(self.words))
        scores, indices = self.faiss_index.search(query_emb, k)
       
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < threshold:
                continue
            results.append({
                'word': self.words[idx],
                'meaning': self.meanings[idx],
                'score': round(float(score), 4)
            })
            if len(results) >= top_k:
                break
        return results

    def search_with_numpy(self, query, top_k=10, threshold=0.3):
        query_emb = self.model.encode(
            query,
            convert_to_tensor=False,
            normalize_embeddings=True
        ).astype(np.float16)
       
        scores = np.dot(self.word_embeddings, query_emb)
       
        if top_k >= len(scores):
            top_indices = np.argsort(-scores)
        else:
            top_indices = np.argpartition(-scores, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]
       
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < threshold:
                break
            results.append({
                'word': self.words[idx],
                'meaning': self.meanings[idx],
                'score': round(score, 4)
            })
            if len(results) >= top_k:
                break
        return results

# ==================== Flask 应用 ====================
app = Flask(__name__)

searcher = None

def get_searcher():
    global searcher
    if searcher is None:
        searcher = EfficientSemanticSearcher()
    return searcher

@app.route('/')
def index():
    s = get_searcher()
    return render_template(
        'index.html',
        total_words=f"{len(s.words):,}",
        model_name='paraphrase-multilingual-MiniLM-L12-v2'
    )

@app.route('/search', methods=['POST'])
def search():
    start_time = time.time()
    
    data = request.get_json()
    query = data.get('query', '').strip()
    top_k = int(data.get('top_k', 10))
    threshold = float(data.get('threshold', 0.3))
    
    if not query:
        return jsonify({'error': '请输入查询内容'}), 400
    
    s = get_searcher()
    results = s.search(query, top_k=top_k, threshold=threshold)
    
    search_time_ms = round((time.time() - start_time) * 1000, 1)
    
    return jsonify({
        'results': results,
        'count': len(results),
        'time_ms': search_time_ms
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)