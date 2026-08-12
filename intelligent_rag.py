"""Intelligent Documentation RAG engine for technical documentation."""
from __future__ import annotations
import argparse, csv, hashlib, html, json, os, re, sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional
try:
    import numpy as np
except ImportError:
    np = None
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError:
    SentenceTransformer = CrossEncoder = None
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None
try:
    from docx import Document
except ImportError:
    Document = None
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
try:
    from fastapi import FastAPI
    from pydantic import BaseModel, Field
except ImportError:
    FastAPI = None
    BaseModel = object
    Field = lambda default=None, **_: default

SUPPORTED = {'.pdf','.docx','.txt','.md','.markdown','.html','.htm','.json','.csv'}
EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

@dataclass
class Record:
    text: str
    source: str
    page: Optional[int] = None
    section: Optional[str] = None
    software: Optional[str] = None
    version: Optional[str] = None
    module: Optional[str] = None
    chunk_id: str = ''
    metadata: dict[str, Any] | None = None

@dataclass
class Result:
    chunk_id: str
    text: str
    score: float
    source: str
    page: Optional[int]
    section: Optional[str]
    software: Optional[str]
    version: Optional[str]
    module: Optional[str]
    metadata: dict[str, Any]

def norm(text: str) -> str:
    text = html.unescape(text or '').replace('\r\n','\n').replace('\r','\n')
    return re.sub(r'\n{3,}','\n\n',re.sub(r'[ \t]+',' ',text)).strip()

def sid(*parts: str) -> str:
    return hashlib.sha256('\x1f'.join(parts).encode()).hexdigest()[:24]

def detect(text: str):
    software = None
    for pattern,name in [(r'\bSchr[oö]dinger\b','Schrödinger'),(r'\bORCA\b','ORCA'),(r'\bMaterial\s+Studio\b','Material Studio'),(r'\bHYSYS\b','HYSYS'),(r'\bJaguar\b','Jaguar'),(r'\bMaestro\b','Maestro'),(r'\bDesmond\b','Desmond')]:
        if re.search(pattern,text,re.I): software=name; break
    version=None
    for p in [r'\b(?:version|ver\.?|release)\s*[:=]?\s*(\d+(?:\.\d+){1,3})\b',r'\b(20\d{2}\.\d+(?:\.\d+)?)\b',r'\bv(\d+(?:\.\d+){1,3})\b']:
        m=re.search(p,text,re.I)
        if m: version=m.group(1); break
    modules=['Maestro','Jaguar','Desmond','Materials Science','QSite','Prime','Epik','LigPrep','Glide','MacroModel','ORCA','HYSYS']
    module=next((m for m in modules if m.lower() in text.lower()),None)
    return software,version,module

def read_file(path: Path):
    ext=path.suffix.lower()
    if ext not in SUPPORTED: return []
    if ext=='.pdf':
        if PdfReader is None: raise RuntimeError('Install pypdf for PDF ingestion')
        return [(norm(p.extract_text() or ''),i+1,None) for i,p in enumerate(PdfReader(str(path)).pages)]
    if ext=='.docx':
        if Document is None: raise RuntimeError('Install python-docx for DOCX ingestion')
        doc=Document(str(path)); out=[]; section=None; buf=[]
        for p in doc.paragraphs:
            text=norm(p.text); style=(p.style.name or '').lower() if p.style else ''
            if not text: continue
            if 'heading' in style:
                if buf: out.append(('\n'.join(buf),None,section)); buf=[]
                section=text
            else: buf.append(text)
        if buf: out.append(('\n'.join(buf),None,section))
        return out
    raw=path.read_text(encoding='utf-8',errors='ignore')
    if ext in {'.html','.htm'}:
        if BeautifulSoup:
            soup=BeautifulSoup(raw,'html.parser')
            for tag in soup(['script','style','noscript']): tag.decompose()
            raw=soup.get_text('\n')
        else: raw=re.sub(r'<[^>]+>',' ',raw)
    elif ext=='.json': raw=json.dumps(json.loads(raw),ensure_ascii=False,indent=2)
    elif ext=='.csv': raw='\n'.join(' | '.join(r) for r in csv.reader(raw.splitlines()))
    return [(norm(raw),None,None)]

def chunk(text: str, size=2600, overlap=350):
    text=norm(text)
    if not text:return []
    paras=re.split(r'\n\s*\n',text); out=[]; cur=''
    for p in paras:
        if not p.strip():continue
        cand=f'{cur}\n\n{p}' if cur else p
        if len(cand)<=size: cur=cand; continue
        if cur: out.append(cur.strip())
        if len(p)<=size:
            tail=cur[-overlap:] if cur else ''
            cur=f'{tail}\n\n{p}'.strip() if tail else p
        else:
            for start in range(0,len(p),size-overlap):
                part=p[start:start+size].strip()
                if part: out.append(part)
                if start+size>=len(p): break
            cur=''
    if cur:out.append(cur.strip())
    return out

class Store:
    def __init__(self,path='rag.sqlite3'):
        self.db=sqlite3.connect(path,check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('''CREATE TABLE IF NOT EXISTS chunks(chunk_id TEXT PRIMARY KEY,text TEXT NOT NULL,source TEXT NOT NULL,page INTEGER,section TEXT,software TEXT,version TEXT,module TEXT,metadata_json TEXT NOT NULL,embedding BLOB)''')
        self.db.commit()
    def put(self,r,e=None):
        blob=np.asarray(e,dtype=np.float32).tobytes() if e is not None and np is not None else None
        self.db.execute('INSERT OR REPLACE INTO chunks VALUES(?,?,?,?,?,?,?,?,?,?)',(r.chunk_id,r.text,r.source,r.page,r.section,r.software,r.version,r.module,json.dumps(r.metadata or {},ensure_ascii=False),blob))
    def commit(self): self.db.commit()
    def records(self):
        rows=self.db.execute('SELECT chunk_id,text,source,page,section,software,version,module,metadata_json FROM chunks').fetchall()
        return [Record(r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[0],json.loads(r[8])) for r in rows]
    def vectors(self):
        if np is None:return {}
        return {a:np.frombuffer(b,dtype=np.float32) for a,b in self.db.execute('SELECT chunk_id,embedding FROM chunks WHERE embedding IS NOT NULL')}

class IntelligentRAG:
    def __init__(self,db_path='rag.sqlite3',embedding_model=EMBED_MODEL,reranker_model=RERANK_MODEL,use_reranker=True):
        self.store=Store(db_path); self.embedding_model=embedding_model; self.reranker_model=reranker_model
        self.embedder=None; self.reranker=None; self.use_reranker=use_reranker; self.records_cache=None; self.bm25=None; self.bm25_ids=[]
    def models(self):
        if SentenceTransformer is None: raise RuntimeError('Install sentence-transformers for semantic retrieval')
        if self.embedder is None:self.embedder=SentenceTransformer(self.embedding_model)
        if self.use_reranker and CrossEncoder is not None and self.reranker is None:self.reranker=CrossEncoder(self.reranker_model)
    def ingest(self,path,software=None,version=None):
        root=Path(path); files=[root] if root.is_file() else [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in SUPPORTED]
        n=0
        for f in files:
            try: blocks=read_file(f)
            except Exception as e: print('[WARN]',f,e); continue
            for text,page,section in blocks:
                ds,dv,dm=detect(f.name+'\n'+text); s=software or ds; v=version or dv
                for i,c in enumerate(chunk(text)):
                    r=Record(c,str(f),page,section,s,v,dm,sid(str(f.resolve()),str(page),str(section),str(i),c),{'filename':f.name,'chunk_index':i})
                    self.store.put(r); n+=1
        self.store.commit(); self.records_cache=None; self.bm25=None; return n
    def build_index(self):
        self.models(); rs=self.store.records()
        if not rs:return
        es=self.embedder.encode([r.text for r in rs],normalize_embeddings=True,show_progress_bar=False)
        for r,e in zip(rs,es):self.store.put(r,e)
        self.store.commit(); self.records_cache=rs
        if BM25Okapi:
            self.bm25_ids=[r.chunk_id for r in rs]; self.bm25=BM25Okapi([self.tokens(r.text) for r in rs])
    @staticmethod
    def tokens(text): return re.findall(r'\w+',text.lower(),re.UNICODE)
    def cache(self):
        if self.records_cache is None:self.records_cache=self.store.records()
        if self.bm25 is None and BM25Okapi and self.records_cache:
            self.bm25_ids=[r.chunk_id for r in self.records_cache]; self.bm25=BM25Okapi([self.tokens(r.text) for r in self.records_cache])
    @staticmethod
    def match(r,s,v,m):
        if s and r.software and r.software.lower()!=s.lower():return False
        if v and r.version and r.version!=v:return False
        if m and r.module and r.module.lower()!=m.lower():return False
        return True
    def semantic(self,q,k,s=None,v=None,m=None):
        self.models(); self.cache(); rs=[r for r in self.records_cache or [] if self.match(r,s,v,m)]
        if not rs:return []
        qv=self.embedder.encode([q],normalize_embeddings=True,show_progress_bar=False)[0]; vs=self.store.vectors(); scored=[]
        for r in rs:
            if r.chunk_id in vs:scored.append((r,float(np.dot(qv,vs[r.chunk_id]))))
        return sorted(scored,key=lambda x:x[1],reverse=True)[:k]
    def lexical(self,q,k,s=None,v=None,m=None):
        self.cache()
        if self.bm25 is None:return []
        by={r.chunk_id:r for r in self.records_cache or []}; scores=self.bm25.get_scores(self.tokens(q)); out=[]
        for cid,score in zip(self.bm25_ids,scores):
            r=by[cid]
            if self.match(r,s,v,m):out.append((r,float(score)))
        return sorted(out,key=lambda x:x[1],reverse=True)[:k]
    @staticmethod
    def rrf(lists,k=60):
        scores={}; rec={}
        for items in lists:
            for rank,(r,_) in enumerate(items,1):scores[r.chunk_id]=scores.get(r.chunk_id,0)+1/(k+rank); rec[r.chunk_id]=r
        return sorted(((rec[c],s) for c,s in scores.items()),key=lambda x:x[1],reverse=True)
    def search(self,q,k=8,candidate_k=30,software=None,version=None,module=None):
        ds,dv,dm=detect(q); software=software or ds; version=version or dv; module=module or dm
        fused=self.rrf([self.semantic(q,candidate_k,software,version,module),self.lexical(q,candidate_k,software,version,module)])[:candidate_k]
        if self.reranker and fused:
            scores=self.reranker.predict([[q,r.text] for r,_ in fused]); fused=sorted(((r,float(s)) for (r,_),s in zip(fused,scores)),key=lambda x:x[1],reverse=True)
        return [Result(r.chunk_id,r.text,s,r.source,r.page,r.section,r.software,r.version,r.module,r.metadata or {}) for r,s in fused[:k]]
    def context(self,q,k=8):
        rs=self.search(q,k)
        if not rs:return 'NO_RELEVANT_DOCUMENTATION_FOUND'
        out=[]
        for i,r in enumerate(rs,1):
            c=f'[{i}] {r.source}'+(f', page {r.page}' if r.page else '')+(f', section: {r.section}' if r.section else '')
            if r.software:c+=f', software: {r.software}'
            if r.version:c+=f', version: {r.version}'
            out.append(f'SOURCE {c}\n{r.text}')
        return '\n\n'.join(out)
    def grounded_prompt(self,q,k=8):
        return f'''You are a documentation-grounded technical assistant.\n\nQUESTION:\n{q}\n\nDOCUMENTATION:\n{self.context(q,k)}\n\nRULES:\n1. Use retrieved documentation as the primary authority.\n2. Never invent parameters, values, commands, syntax, or version behavior.\n3. Preserve exact syntax when documented.\n4. Prefer the requested software version and never silently substitute another.\n5. Cite source numbers for configuration claims.\n6. Clearly label inference.\n7. If evidence is insufficient, say it was not found instead of guessing.\n8. Report conflicts between versions or sources.'''

if FastAPI:
    app=FastAPI(title='Intelligent Documentation RAG',version='1.0')
    class Request(BaseModel):
        query:str; k:int=Field(8,ge=1,le=50); software:Optional[str]=None; version:Optional[str]=None; module:Optional[str]=None
    rag=IntelligentRAG(os.getenv('RAG_DB','rag.sqlite3'),use_reranker=os.getenv('RAG_RERANK','1')!='0')
    @app.get('/health')
    def health():return {'status':'ok'}
    @app.post('/search')
    def api_search(x:Request):return {'results':[asdict(r) for r in rag.search(x.query,x.k,software=x.software,version=x.version,module=x.module)]}
    @app.post('/grounded-prompt')
    def api_prompt(x:Request):return {'prompt':rag.grounded_prompt(x.query,x.k)}
else: app=None

def main():
    p=argparse.ArgumentParser(description='Intelligent Documentation RAG'); sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('ingest'); a.add_argument('path'); a.add_argument('--db',default='rag.sqlite3'); a.add_argument('--software'); a.add_argument('--version')
    a=sub.add_parser('index'); a.add_argument('--db',default='rag.sqlite3'); a.add_argument('--embedding-model',default=EMBED_MODEL); a.add_argument('--no-reranker',action='store_true')
    a=sub.add_parser('search'); a.add_argument('query'); a.add_argument('--db',default='rag.sqlite3'); a.add_argument('-k',type=int,default=8); a.add_argument('--software'); a.add_argument('--version'); a.add_argument('--module')
    a=sub.add_parser('prompt'); a.add_argument('query'); a.add_argument('--db',default='rag.sqlite3'); a.add_argument('-k',type=int,default=8)
    x=p.parse_args()
    if x.cmd=='ingest':print('Ingested',IntelligentRAG(x.db).ingest(x.path,x.software,x.version),'chunks.')
    elif x.cmd=='index':IntelligentRAG(x.db,x.embedding_model,use_reranker=not x.no_reranker).build_index(); print('Index built.')
    elif x.cmd=='search':print(json.dumps([asdict(r) for r in IntelligentRAG(x.db).search(x.query,x.k,software=x.software,version=x.version,module=x.module)],ensure_ascii=False,indent=2))
    elif x.cmd=='prompt':print(IntelligentRAG(x.db).grounded_prompt(x.query,x.k))
if __name__=='__main__':main()
