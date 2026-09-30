#!/usr/bin/env python3
"""IVAP - Intelligent Video Analytics Platform (single file). Start with run.bat / run.sh (see README.md)."""
import os, io, re, json, math, time, uuid, hmac, base64, queue, hashlib, secrets, sqlite3, threading, zipfile, webbrowser, datetime
os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'rtsp_transport;tcp|fflags;nobuffer|flags;low_delay')
from collections import Counter
from functools import wraps
import cv2, numpy as np
# Avoid OpenCV/PyTorch oversubscribing cores on low-power machines.
cv2.setNumThreads(1)
from flask import Flask, request, jsonify, Response, send_file
from ultralytics import YOLO
DEVICE_LABEL = 'CPU'; HALF = False; REQUESTED_DEVICE = os.getenv('IBVAP_DEVICE', 'auto').lower()
try:
    import torch
    torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2) // 2)))
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    xpu = getattr(torch, 'xpu', None)
    if REQUESTED_DEVICE == 'directml':
        import torch_directml
        DEV = torch_directml.device(); DEVICE_LABEL = 'DirectML'; HALF = False
    elif REQUESTED_DEVICE in ('auto', 'cuda') and torch.cuda.is_available():
        DEV = 0; DEVICE_LABEL = 'ROCm' if getattr(torch.version, 'hip', None) else 'CUDA'; HALF = True
    elif REQUESTED_DEVICE in ('auto', 'xpu') and xpu is not None and xpu.is_available():
        DEV = 'xpu'; DEVICE_LABEL = 'Intel XPU'; HALF = True
    else: DEV = 'cpu'
except Exception as e:
    DEV = 'cpu'; DEVICE_LABEL = 'CPU'
    if REQUESTED_DEVICE != 'auto': print(f'GPU backend unavailable; using CPU: {e!r}')
OCR = None      # pip-only RapidOCR first, Tesseract binary as fallback; both return (text, confidence)
try:
    from rapidocr_onnxruntime import RapidOCR; _r = RapidOCR()
    def OCR(img):
        res, _ = _r(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        return (' '.join(x[1] for x in res), sum(float(x[2]) for x in res) / len(res)) if res else ('', 0.0)
except Exception:
    try:
        import pytesseract; pytesseract.get_tesseract_version()
        def OCR(img): return pytesseract.image_to_string(img, config='--psm 7'), -1.0
    except Exception: OCR = None
try: import paho.mqtt.client as mqtt
except Exception: mqtt = None

BASE = os.path.dirname(os.path.abspath(__file__)); D = BASE + '/ivap_data'; MD = BASE + '/models'
DEMO_VIDEO_DIR = BASE + '/demo_videos'
for p in (D + '/snaps', D + '/videos', MD, DEMO_VIDEO_DIR): os.makedirs(p, exist_ok=True)
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
YOLO_PATH = MD + '/yolo11n.pt'; TRACKER_PATH = BASE + '/ivap_bytetrack.yaml'; MODEL_READY = threading.Event(); MODEL_ERROR = None; Z = '0' * 64; L = threading.RLock()
NAMES = {0: 'person', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'car'}
RED, GREEN, BLUE, YELLOW = (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)   # BGR
RL = {'viewer': 1, 'operator': 2, 'admin': 3}
db = sqlite3.connect(D + '/ivap.db', check_same_thread=False); db.row_factory = sqlite3.Row
db.executescript('''create table if not exists users(u text primary key,h text,role text);
create table if not exists cams(id integer primary key,name text,src text,fence text,fx real,fy real);
create table if not exists fences(id integer primary key autoincrement,cam int not null,name text not null,poly text not null);
create table if not exists alerts(id integer primary key,ts real,cam int,type text,sev text,info text,snap text,inc int,prev text,hash text);
create table if not exists incidents(id integer primary key,ts real,cam int,title text,status text);
create table if not exists rules(id integer primary key,etype text,cam int,sev text,incident int,cnt int default 1,win int default 60);
create table if not exists ledger(id integer primary key,ts real,ah text,prev text,hash text);
create table if not exists reads(id integer primary key,ts real,cam int,track int,raw text,conf real);
create table if not exists plates(id integer primary key,ts real,cam int,track int,plate text,votes int);
create table if not exists daily_tracks(day text not null,cam int not null,session text not null,track int not null,kind text not null,first_ts real not null,primary key(day,cam,session,track));
create table if not exists settings(k text primary key,v text)''')
db.execute('create index if not exists idx_alerts_ts on alerts(ts)'); db.commit()
db.execute('create index if not exists idx_daily_tracks_day on daily_tracks(day,kind)'); db.commit()
for t_, col in (('rules', 'cnt int default 1'), ('rules', 'win int default 60'), ('reads', 'crop text')):
    try: db.execute(f'alter table {t_} add column ' + col)
    except sqlite3.OperationalError: pass
# Face recognition is intentionally removed, including any enrolled face data.
db.execute('drop table if exists faces'); db.commit()

def q(sql, a=(), w=False):
    with L:
        c = db.execute(sql, a)
        if w: db.commit(); return c.lastrowid
        return [dict(r) for r in c.fetchall()]
def qmany(sql, rows):
    if not rows: return
    with L:
        db.executemany(sql, rows); db.commit()
DAILY_LIMIT_KEYS=('person','car','bike','bus')
def read_daily_limits():
    limits={k:0 for k in DAILY_LIMIT_KEYS}; saved=q("select v from settings where k='daily_limits'")
    if not saved: return limits
    try:
        values=json.loads(saved[0]['v'])
        for key in DAILY_LIMIT_KEYS:
            value=values.get(key,0)
            if not isinstance(value,bool) and str(value).strip().lstrip('+').isdigit(): limits[key]=max(0,min(1000000,int(value)))
    except Exception: pass
    return limits
# Migrate each former single-camera polygon into the new named fence collection.
for old in q('select id,fence from cams where fence is not null'):
    if not q('select id from fences where cam=?', (old['id'],)):
        try:
            pts=json.loads(old['fence'])
            if len(pts)>=3: q('insert into fences(cam,name,poly) values(?,?,?)', (old['id'],'Fence 1',json.dumps(pts)), True)
        except Exception: pass
    q('update cams set fence=null where id=?', (old['id'],), True)
def sha(s): return hashlib.sha256(s.encode()).hexdigest()
def hh(p, ts, cam, typ, sev, info, snap): return sha(f'{p}|{ts}|{cam}|{typ}|{sev}|{info}|{snap}')
def ph(p, salt=None):
    salt = salt or secrets.token_hex(8)
    return salt + '$' + hashlib.pbkdf2_hmac('sha256', p.encode(), salt.encode(), 100000).hex()

if not q("select 1 from settings where k='secret'"): q("insert into settings values('secret',?)", (secrets.token_hex(32),), True)
SECRET = q("select v from settings where k='secret'")[0]['v'].encode()
if not q('select 1 from users'): q('insert into users values(?,?,?)', ('admin', ph('admin123'), 'admin'), True)
CFG = {'ai_fps': 30.0, 'dwell_s': 10.0, 'imgsz': 320, 'stream_w': 960, 'anpr': 1, 'reid': 0, 'mqtt_host': os.getenv('MQTT_HOST', ''), 'mqtt_port': int(os.getenv('MQTT_PORT', 1883))}
_c = q("select v from settings where k='cfg'")
if _c: CFG.update(json.loads(_c[0]['v']))
CFG.pop('face', None)
if not q("select 1 from settings where k='perf_tune_v3'"):
    CFG['imgsz'] = min(320, int(CFG.get('imgsz', 320)))
    CFG['stream_w'] = min(960, int(CFG.get('stream_w', 960)))
    CFG['reid'] = 0
    q("insert or replace into settings values('cfg',?)", (json.dumps(CFG),), True)
    q("insert into settings values('perf_tune_v3','1')", (), True)
CFG['ai_fps'] = min(120.0, max(15.0, float(CFG.get('ai_fps', 30.0))))

# ---------- HS256 JWT ----------
def b64(b): return base64.urlsafe_b64encode(b).rstrip(b'=').decode()
def sig(m): return b64(hmac.new(SECRET, m.encode(), 'sha256').digest())
def mk(u, role):
    m = b64(b'{"alg":"HS256","typ":"JWT"}') + '.' + b64(json.dumps({'u': u, 'r': role, 'exp': time.time() + 8 * 3600}).encode()); return m + '.' + sig(m)
def chk(t):
    try:
        h, p, s = t.split('.'); assert hmac.compare_digest(s, sig(h + '.' + p))
        d = json.loads(base64.urlsafe_b64decode(p + '=' * (-len(p) % 4))); assert d['exp'] > time.time(); return d
    except Exception: return None
def auth(lvl=1):
    def d(f):
        @wraps(f)
        def w(*a, **k):
            u = chk(request.headers.get('Authorization', '')[7:] or request.args.get('t', ''))
            if not u: return jsonify(error='auth'), 401
            if RL.get(u['r'], 0) < lvl: return jsonify(error='forbidden'), 403
            request.u = u; return f(*a, **k)
        return w
    return d

# ---------- MQTT hook (configurable in the UI) ----------
MQ = None
def mq_connect():
    global MQ
    if MQ:
        try: MQ.loop_stop(); MQ.disconnect()
        except Exception: pass
        MQ = None
    if mqtt and CFG['mqtt_host']:
        try:
            try: c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            except AttributeError: c = mqtt.Client()
            c.connect_async(CFG['mqtt_host'], int(CFG['mqtt_port'])); c.loop_start(); MQ = c    # auto-reconnects
        except Exception as e: print('mqtt off:', e)

# ---------- alerts: rule engine, incidents, hash chain, ledger ----------
def save_img(name, img):
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if ok:
        with open(os.path.join(D, 'snaps', name), 'wb') as fh: fh.write(buf.tobytes())
    return bool(ok)

def alert(cam, typ, sev, info, frame=None):
    ts = time.time(); inc = 0
    for r in q('select * from rules where etype=? and (cam is null or cam=?)', (typ, cam)):
        n = q('select count(*) c from alerts where cam=? and type=? and ts>?', (cam, typ, ts - (r['win'] or 60)))[0]['c'] + 1
        if n >= (r['cnt'] or 1): sev = r['sev'] or sev; inc |= r['incident'] or 0
    snap = ''
    if frame is not None:
        n = uuid.uuid4().hex + '.jpg'
        try:
            if save_img(n, frame): snap = n
        except Exception as e: print('evidence save failed:', e)
    with L:
        p = (q('select hash from alerts order by id desc limit 1') or [{'hash': Z}])[0]['hash']; iid = None
        if inc or sev in ('high', 'critical'):
            o = q("select id from incidents where status='open' and cam=? and ts>?", (cam, ts - 300)); nm = CAMS[cam].r['name'] if cam in CAMS else cam
            iid = o[0]['id'] if o else q('insert into incidents(ts,cam,title,status) values(?,?,?,?)', (ts, cam, f'{typ} @ {nm}', 'open'), True)
        h = hh(p, ts, cam, typ, sev, info, snap)
        aid = q('insert into alerts(ts,cam,type,sev,info,snap,inc,prev,hash) values(?,?,?,?,?,?,?,?,?)', (ts, cam, typ, sev, info, snap, iid, p, h), True)
        lp = (q('select hash from ledger order by id desc limit 1') or [{'hash': Z}])[0]['hash']
        q('insert into ledger(ts,ah,prev,hash) values(?,?,?,?)', (ts, h, lp, sha(f'{lp}|{h}|{ts}')), True)
    if MQ:
        try: MQ.publish('ivap/alerts', json.dumps(dict(id=aid, cam=cam, type=typ, sev=sev, info=info, ts=ts)))
        except Exception: pass

# ---------- background workers: auxiliary models never hold up YOLO ----------
JOBS = queue.Queue(maxsize=128)
def submit(fn, *a):
    try: JOBS.put_nowait((fn, a)); return True
    except queue.Full: return False
def worker():
    while True:
        fn, a = JOBS.get()
        try:
            fn(*a)
        except Exception as e: print('job error:', repr(e))

PLATE_RE = re.compile(r'^(?=.*[A-Z])(?=.*[0-9])[A-Z0-9]{5,12}$')
PLATE_IN_RE = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$')
AMB_L2D = str.maketrans({'O':'0','Q':'0','D':'0','I':'1','L':'1','Z':'2','S':'5','G':'6','T':'7','B':'8'})
AMB_D2L = str.maketrans({'0':'O','1':'I','2':'Z','5':'S','6':'G','8':'B'})

def plate_norms(raw):
    s = re.sub(r'[^A-Z0-9]', '', str(raw).upper())
    if not 5 <= len(s) <= 12: return []
    if PLATE_IN_RE.fullmatch(s): return [s]
    out = [s]
    # Indian registration layout: state, district, series, number. Keep generic
    # OCR text too, so the app remains usable with other plate formats.
    for dl in (1, 2):
        for sl in (1, 2, 3):
            fl = len(s) - 2 - dl - sl
            if not 1 <= fl <= 4: continue
            cand = (s[:2].translate(AMB_D2L) + s[2:2+dl].translate(AMB_L2D) +
                    s[2+dl:2+dl+sl].translate(AMB_D2L) + s[-fl:].translate(AMB_L2D))
            if PLATE_IN_RE.fullmatch(cand): out.append(cand)
    return list(dict.fromkeys(out))

def plate_plausibility(text):
    return 1.0 if PLATE_IN_RE.fullmatch(text) else (.55 if PLATE_RE.fullmatch(text) else 0.0)

def smooth_box(raw, previous, now):
    """Track-ID box smoother: bounded constant-velocity prediction plus EMA size smoothing."""
    raw = [float(x) for x in raw]
    if previous is None or now - previous['last'] > 2.0:
        return raw, 0.0, 0.0
    dt = max(.001, min(.8, now - previous['last'])); old = previous['box']
    pcx, pcy = (old[0]+old[2])/2, (old[1]+old[3])/2
    rvx = max(-2000., min(2000., ((raw[0]+raw[2])/2-pcx)/dt)); rvy = max(-2000., min(2000., ((raw[1]+raw[3])/2-pcy)/dt))
    vx = .5*previous['vx'] + .5*rvx; vy = .5*previous['vy'] + .5*rvy
    pred = [old[i] + (previous['vx'] if i in (0,2) else previous['vy'])*dt for i in range(4)]
    # Favor the current detector measurement to reduce visible lag while keeping
    # a small predicted component to damp frame-to-frame detector jitter.
    box = [.2*pred[i] + .8*raw[i] for i in range(4)]
    ow,oh=max(4.,old[2]-old[0]),max(4.,old[3]-old[1]); w,h=max(4.,box[2]-box[0]),max(4.,box[3]-box[1])
    w,h=.45*ow+.55*w,.45*oh+.55*h; cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
    return [cx-w/2,cy-h/2,cx+w/2,cy+h/2],vx,vy

def predict_box(previous, now, max_horizon=.8):
    """Bounded short-term box prediction used while a tracker reacquires a target."""
    dt = max(0.0, min(max_horizon, now - previous['last']))
    dx = max(-400.0, min(400.0, previous['vx'] * dt))
    dy = max(-400.0, min(400.0, previous['vy'] * dt))
    x1, y1, x2, y2 = previous['box']
    return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]

def assign_track_id(box, cls, raw_id, now, tracks, claimed, next_id):
    """Keep a stable local ID when detector tracker IDs flicker or disappear."""
    x1,y1,x2,y2 = map(float, box); bw=max(1.,x2-x1); bh=max(1.,y2-y1); cx=(x1+x2)/2; cy=(y1+y2)/2
    best_id=None; best_score=-1.0
    for tid,st in tracks.items():
        if tid in claimed or tracking_group(cls)!=tracking_group(st['cls']): continue
        age=max(0.,now-st['last']); p=predict_box(st,now); px1,py1,px2,py2=p
        ix=max(0.,min(x2,px2)-max(x1,px1)); iy=max(0.,min(y2,py2)-max(y1,py1)); inter=ix*iy
        union=bw*bh+max(1.,px2-px1)*max(1.,py2-py1)-inter; iou=inter/max(1.,union)
        pcx=(px1+px2)/2; pcy=(py1+py2)/2; dist=math.hypot(cx-pcx,cy-pcy)/max(24.,math.sqrt(bw*bh))
        same_raw=raw_id>=0 and st.get('raw_id')==raw_id
        if same_raw and age<=2.0: score=2.0+iou
        elif age<=1.5 and (iou>=.05 or dist<=1.1): score=iou+.35*max(0.,1.-dist)+(.08 if st['cls']==cls else 0.)
        else: continue
        if score>best_score: best_id,best_score=tid,score
    if best_id is None: best_id=next_id; next_id+=1
    claimed.add(best_id)
    return best_id,next_id

def tracking_group(cls):
    return 'car' if int(cls) in (2,7) else int(cls)

def dedupe_car_aliases(boxes):
    """Suppress near-identical car/truck-class boxes before they become separate car tracks."""
    candidates=[]
    for b in boxes:
        coords=tuple(map(float,b.xyxy[0].tolist())); cls=int(b.cls[0]); raw_id=int(b.id[0]) if b.id is not None else -1
        conf=float(b.conf[0]) if getattr(b,'conf',None) is not None else 0.0
        candidates.append((coords,cls,raw_id,conf))
    candidates.sort(key=lambda x:x[3],reverse=True); kept=[]
    for item in candidates:
        (x1,y1,x2,y2),cls,_,_=item; duplicate=False
        if cls in (2,7):
            area=max(1.,(x2-x1)*(y2-y1))
            for (a,b,c,d),other_cls,_,_ in kept:
                if other_cls not in (2,7): continue
                ix=max(0.,min(x2,c)-max(x1,a)); iy=max(0.,min(y2,d)-max(y1,b)); inter=ix*iy
                other_area=max(1.,(c-a)*(d-b)); iou=inter/max(1.,area+other_area-inter); contained=inter/min(area,other_area)
                if iou>=.82 or (other_cls!=cls and contained>=.92): duplicate=True; break
        if not duplicate: kept.append(item)
    return [((int(x1),int(y1),int(x2),int(y2)),cls,raw_id) for (x1,y1,x2,y2),cls,raw_id,_ in kept]

def plate_reads(crop):
    """Return deduplicated multi-pass OCR candidates with original-resolution crops."""
    if not OCR or crop is None or crop.size == 0: return []
    if crop.shape[1] > 900: crop = cv2.resize(crop, None, fx=900/crop.shape[1], fy=900/crop.shape[1], interpolation=cv2.INTER_AREA)
    gray = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8,8)).apply(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    smooth = cv2.bilateralFilter(gray, 7, 55, 55); edges = cv2.Canny(smooth, 45, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE); h0,w0=gray.shape[:2]; best={}
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        x,y,w,h=cv2.boundingRect(c); area=w*h
        if h < 8 or w < 45 or not 2.0 < w/max(1,h) < 6.8 or area < max(100,.00035*w0*h0): continue
        px,py=max(2,int(.04*w)),max(2,int(.12*h)); x0,y0=max(0,x-px),max(0,y-py); x1,y1=min(w0,x+w+px),min(h0,y+h+py)
        src=gray[y0:y1,x0:x1]; scale=max(2.5,min(5.,220./max(1,src.shape[1]))); roi=cv2.resize(src,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
        otsu=cv2.threshold(roi,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        variants=(roi,otsu)
        for variant in variants:
            with OCR_LOCK: txt,cf=OCR(variant)
            cf=float(cf) if cf is not None and float(cf)>=0 else .45
            for norm in plate_norms(txt):
                score=cf+.25*plate_plausibility(norm)
                if norm not in best or score>best[norm][0]: best[norm]=(score,cf,crop[y0:y1,x0:x1])
    return [(text,round(v[1],2),v[2]) for text,v in sorted(best.items(),key=lambda item:item[1][0],reverse=True)[:6]]

def vote(readings):
    """Confidence-weighted character consensus; accepts strings or (text, confidence)."""
    vals=[]
    for item in readings:
        text,cf=(item[0],item[1]) if isinstance(item,tuple) else (item,.45)
        options=[x for x in plate_norms(text) if plate_plausibility(x)]
        if options: vals.append((max(options,key=plate_plausibility),max(.1,float(cf))))
    if len(vals)<3: return None
    length=Counter(len(x) for x,_ in vals).most_common(1)[0][0]; group=[v for v in vals if len(v[0])==length]
    if len(group)<3: return None
    result=''
    for i in range(length):
        scores={}
        for text,cf in group: scores[text[i]]=scores.get(text[i],0)+cf
        result+=max(scores,key=scores.get)
    return result if plate_plausibility(result)>0 else None

PLT = {}; PLT_LOCK = threading.RLock(); OCR_LOCK = threading.Lock()
GAL = []; GL = threading.Lock()
def reid(cr, cam):
    hsv = cv2.cvtColor(cr, cv2.COLOR_BGR2HSV); n = hsv.shape[0] // 2
    v = np.concatenate([cv2.normalize(cv2.calcHist([p], [0, 1], None, [8, 8], [0, 180, 0, 256]), None, norm_type=cv2.NORM_L2).flatten() for p in (hsv[:n], hsv[n:])])
    v /= np.linalg.norm(v) + 1e-9; now = time.time()
    with GL:
        best = max(((float(v @ e['v']), e) for e in GAL if now - e['ts'] < 3600), key=lambda x: x[0], default=(0, None))
        if best[0] > .92:
            e = best[1]; old = e['cam']; e['v'] = .7 * e['v'] + .3 * v; e['v'] /= np.linalg.norm(e['v']); e['cam'] = cam; e['ts'] = now
            return e['g'], (old if old != cam else None)
        GAL.append(dict(g=len(GAL) + 1, v=v, cam=cam, ts=now)); return len(GAL), None

# ---------- camera: reader thread (video at source rate) + AI thread (own rate) ----------
CAMS = {}
class Cam:
    def __init__(s, r):
        s.r = r; s.id = r['id']; s.go = True; s.simulation = bool(r.get('simulation', False)); s.anpr_enabled = bool(r.get('anpr_enabled', not s.simulation)); s.status = 'connecting'; s.ai_state = 'loading'; s.ai_ready = False; s.m = None; s.frame = None; s.ai_frame = None; s.ai_frame_ts = 0; s.fid = 0; s.W = s.H = 0
        s.jpg = None; s.n = 0; s.stream_lock = threading.Lock(); s.track_lock = threading.RLock(); s.ai_seq = 0; s.stream_event = threading.Event(); s.viewers = 0; s.tlast = 0; s.det_updated_at = 0.0; s.cfps = s.afps = s.alast = 0.0; s.dets = []; s.ev = []; s.hist = {}; s.tracks = {}; s.track_seen = {}; s.ims = 0.0
        s.fence_lock=threading.RLock(); s.fences = {x['id']:{'name':x['name'],'poly':np.array(json.loads(x['poly']),np.int32)} for x in q('select * from fences where cam=? order by id',(s.id,))}
        s.track_session=uuid.uuid4().hex; s.daily_day=''; s.daily_seen=set()
        s.cool = {}; s.din = {}; s.loitered = set(); s.rt = {}; s.gid = {}; s.an = {}; s.anpr_pending = set(); s.next_tid = 1; s.prev = None; s.still = 0; s.tp = ''
    def begin(s):
        threading.Thread(target=s.read, daemon=True).start()
        threading.Thread(target=s.ai, daemon=True).start()
        threading.Thread(target=s.stream_encode, daemon=True).start()
    def ok(s, k, sec):
        n = time.time()
        if n - s.cool.get(k, 0) > sec: s.cool[k] = n; return True
        return False
    def setst(s, st):
        if st != s.status:
            old = s.status; s.status = st
            if not s.simulation:
                if st == 'no signal': alert(s.id, 'signal_loss', 'high', 'camera signal lost')
                elif old == 'no signal': alert(s.id, 'signal_restored', 'info', 'camera signal restored')
    def render(s, f, tf, dets=None):       # render the exact frame YOLO analyzed with its matching boxes
        g = f.copy()
        with s.fence_lock: fence_polys=[z['poly'] for z in s.fences.values()]
        for poly in fence_polys: cv2.polylines(g, [poly], True, YELLOW, 2)
        for x1, y1, x2, y2, lab, col, vx, vy, ts in (s.dets if dets is None else dets):
            x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(g.shape[1] - 1, x2), min(g.shape[0] - 1, y2)
            cv2.rectangle(g, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA); cv2.putText(g, lab, (x1, max(16, y1 - 6)), 0, .55, col, 2, cv2.LINE_AA)
        if s.tp: cv2.putText(g, 'TAMPER: ' + s.tp, (10, 30), 0, .8, RED, 2)
        cv2.putText(g, f'{s.cfps:.0f} fps | AI {s.afps:.0f}', (8, g.shape[0] - 10), 0, .5, (255, 255, 255), 1)
        w = int(CFG['stream_w'])
        return cv2.resize(g, (w, int(g.shape[0] * w / g.shape[1]))) if g.shape[1] > w else g
    def shot(s): return s.render(s.ai_frame, s.ai_frame_ts, s.dets) if s.ai_frame is not None else None
    def fire(s, typ, sev, info): alert(s.id, typ, sev, info, s.shot())

    def read(s):
        cap = None; miss = 0; src = s.r['src']
        if src.startswith('browser:'):
            while s.go:
                if s.frame is None: s.setst('connecting')
                elif time.time() - s.tlast > 2: s.setst('no signal')
                else: s.setst('online')
                s.stream_event.wait(.1); s.stream_event.clear()
            return
        if not src.isdigit() and '://' not in src and not os.path.isabs(src): src = os.path.join(BASE, src)
        isf = os.path.isfile(src); dt = 0
        while s.go:
            try:
                if cap is None:
                    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
                    if not cap.isOpened(): cap.release(); cap = None; s.setst('no signal'); time.sleep(3); continue    # auto-reconnect
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1); miss = 0; dt = 1 / (cap.get(cv2.CAP_PROP_FPS) or 25) if isf else 0
                t0 = time.time(); ok, f = cap.read()
                if not ok:
                    if isf:
                        # Seek first, then reopen as a fallback for codecs that fail to rewind reliably.
                        try: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ok, f = cap.read()
                        except Exception: ok = False
                        if not ok:
                            cap.release(); cap=cv2.VideoCapture(src)
                            if cap.isOpened():
                                cap.set(cv2.CAP_PROP_POS_FRAMES,0); ok,f=cap.read()
                            if not ok:
                                cap.release(); cap=None; time.sleep(.05); continue
                    else:
                        miss += 1; time.sleep(.05)
                        if miss >= 20: cap.release(); cap = None; s.setst('no signal'); time.sleep(2)
                        continue
                miss = 0; s.setst('online'); t1 = time.time()
                if s.tlast: s.cfps = .9 * s.cfps + .1 / max(1e-3, t1 - s.tlast) if s.cfps else 1 / max(1e-3, t1 - s.tlast)     # measured capture FPS
                s.tlast = t1; s.H, s.W = f.shape[:2]; s.frame = f; s.fid += 1
                s.stream_event.set()
                if isf: time.sleep(max(0, dt - (time.time() - t0)))                      # prerecorded files play at native speed
            except Exception as e: print(f'cam {s.id} reader:', repr(e)); time.sleep(1)
        if cap: cap.release()

    def ai(s):
        seen = -1; last = 0
        while s.go:
            try:
                if s.m is None:
                    if not MODEL_READY.is_set(): s.ai_state = 'loading model weights'; time.sleep(.05); continue
                    if MODEL_ERROR: s.ai_state = 'error'; time.sleep(1); continue
                    s.ai_state = 'loading model'
                    s.m = YOLO(YOLO_PATH)
                    s.ai_state = 'waiting for source'
                now = time.time()
                if s.frame is not None and now - s.tlast > 5: s.setst('no signal')     # watchdog for stalled streams
                if s.frame is None or s.fid == seen or s.status != 'online' or now - last < 1 / max(15, CFG['ai_fps']): time.sleep(.001); continue
                f = s.frame; ft = s.tlast; seen = s.fid; last = now; s.ev = []
                out = s.proc(f, ft)
                if not s.simulation: s.tamper(f)
                s.dets = out
                t = time.time()
                s.det_updated_at = t
                if s.alast: s.afps = .8 * s.afps + .2 / max(1e-3, t - s.alast) if s.afps else 1 / max(1e-3, t - s.alast)    # measured AI FPS
                s.alast = t
                s.ai_frame = f; s.ai_frame_ts = ft; s.ai_seq += 1; s.ai_state = 'ready'; s.ai_ready = True
                if s.ev:
                    processed = s.render(f, ft, out)
                    for e in s.ev: alert(s.id, *e, processed)
            except Exception as e:
                s.ai_state = 'error'; print(f'cam {s.id} ai:', repr(e)); time.sleep(1)

    def stream_encode(s):
        seen = 0; encoded_at = 0.0
        while s.go:
            if not s.stream_event.wait(.1): continue
            s.stream_event.clear(); seq = s.fid; frame = s.frame
            if not s.viewers or not s.ai_ready or frame is None or seq == seen: continue
            try:
                now = time.time()
                if now - encoded_at < 1 / 30: continue
                width = int(CFG['stream_w'])
                if width > 0 and frame.shape[1] > width: frame = cv2.resize(frame, (width, int(frame.shape[0]*width/frame.shape[1])), interpolation=cv2.INTER_AREA)
                enc = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])[1]
                if enc is not None:
                    with s.stream_lock: s.jpg = enc.tobytes(); s.n += 1
                    seen = seq; encoded_at = time.time()
            except Exception as e: print(f'cam {s.id} stream encoder:', repr(e))

    def proc(s, f, now):
        t0 = time.time()
        r = s.m.track(f, persist=True, tracker=TRACKER_PATH, classes=list(NAMES), conf=.20, imgsz=int(CFG['imgsz']), max_det=80, half=HALF, verbose=False, device=DEV)[0]
        ms = (time.time() - t0) * 1000; s.ims = .8 * s.ims + .2 * ms if s.ims else ms; out = []; seen_ids = set(); daily_rows=[]
        day_key=datetime.datetime.fromtimestamp(now).date().isoformat()
        if day_key != s.daily_day: s.daily_day=day_key; s.daily_seen.clear()
        claimed_ids = set()
        for (x1,y1,x2,y2),c,raw_t in dedupe_car_aliases(r.boxes):
            t,s.next_tid = assign_track_id((x1,y1,x2,y2),c,raw_t,now,s.tracks,claimed_ids,s.next_tid)
            col = GREEN if c == 0 else BLUE; lab = NAMES[c] + (f' #{t}' if t >= 0 else ''); vx = vy = 0.0
            if t >= 0:
                seen_ids.add(t); sb, vx, vy = smooth_box((x1,y1,x2,y2), s.tracks.get(t), now)
                with s.track_lock:
                    s.tracks[t] = {'box': sb, 'vx': vx, 'vy': vy, 'last': now, 'cls': c, 'lab': lab, 'raw_id': raw_t}
                    s.track_seen[t] = now
                if not s.simulation and t not in s.daily_seen:
                    s.daily_seen.add(t); kind={0:'person',2:'car',3:'bike',5:'bus',7:'car'}.get(c)
                    if kind: daily_rows.append((day_key,s.id,s.track_session,t,kind,now))
                x1, y1, x2, y2 = map(int, np.round(sb)); s.hist[t] = ((x1+x2)/2, (y1+y2)/2, now, vx, vy)
            x1, y1, x2, y2 = max(0,x1), max(0,y1), min(f.shape[1],x2), min(f.shape[0],y2)
            cr = f[y1:y2, x1:x2]
            # The bottom-center of the smoothed box is the common point used for zone decisions.
            active_zones=set()
            if t >= 0:
                with s.fence_lock:
                    for fid,zone in list(s.fences.items()):
                        inside=cv2.pointPolygonTest(zone['poly'],(float((x1+x2)/2),float(y2)),False)>=0; key=(fid,t)
                        if inside:
                            active_zones.add(fid); entered=s.din.setdefault(key,now)
                            if entered==now: s.ev.append(('intrusion','high',f'{NAMES[c]} #{t} entered {zone["name"]}'))
                            if now-entered>=CFG['dwell_s'] and key not in s.loitered:
                                s.loitered.add(key); s.ev.append(('loitering','medium',f'{NAMES[c]} #{t} dwelling in {zone["name"]} for {int(now-entered)}s'))
                        elif key in s.din:
                            entered=s.din.pop(key); s.loitered.discard(key)
                            s.ev.append(('fence_exit','info',f'{NAMES[c]} #{t} left {zone["name"]} after {int(max(0,now-entered))}s'))
            if active_zones: col=RED
            if t >= 0 and cr.size:
                if c == 0 and y2 - y1 > 40:
                    if CFG['reid'] and not s.simulation and now - s.rt.get(t, 0) > 3: s.rt[t] = now; submit(s.reid_job, t, cr)
                    lab += (f' G{s.gid[t]}' if t in s.gid else '')
                elif c != 0 and OCR and s.anpr_enabled and CFG['anpr'] and x2 - x1 > 80:
                    # Throttle OCR by YOLO frame count; YOLO itself still runs on every eligible frame.
                    s.an[t] = s.an.get(t, 0) + 1
                    if s.an[t] % 6 == 0 and s.an[t] <= 96 and t not in s.anpr_pending:
                        s.anpr_pending.add(t)
                        if not submit(s.run_anpr_job, t, cr.copy()): s.anpr_pending.discard(t)
                    lab += ' ' + (PLT.get((s.id, t)) or {}).get('v', '')
            out.append((x1, y1, x2, y2, lab, col, vx, vy, now))
        for tid, st in list(s.tracks.items()):
            missed = now - st['last']
            if tid not in seen_ids and missed <= 1.0:
                a,b,c,d = map(int, np.round(predict_box(st, now))); a,b=max(0,min(f.shape[1],a)),max(0,min(f.shape[0],b)); c,d=max(0,min(f.shape[1],c)),max(0,min(f.shape[0],d))
                with s.fence_lock: fence_polys=[z['poly'] for z in s.fences.values()]
                col = RED if any(cv2.pointPolygonTest(poly,(float((a+c)/2),float(d)),False)>=0 for poly in fence_polys) else (GREEN if st['cls']==0 else BLUE)
                out.append((a,b,c,d,st['lab'], col, st['vx'],st['vy'],now))
            elif missed > 2.0:
                with s.track_lock: s.tracks.pop(tid, None)
                with s.fence_lock:
                    for key in [k for k in s.din if k[1]==tid]: s.din.pop(key,None); s.loitered.discard(key)
        if daily_rows: qmany('insert or ignore into daily_tracks(day,cam,session,track,kind,first_ts) values(?,?,?,?,?,?)',daily_rows)
        if len(s.hist) > 300: s.hist = {k: v for k, v in s.hist.items() if now - v[2] < 10}
        with s.track_lock:
            if len(s.track_seen) > 500: s.track_seen = {k: v for k, v in s.track_seen.items() if now - v < 10}
        return out

    def tamper(s, f):
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (64, 48)).astype(np.float32); p = s.prev; s.prev = g; k = ''
        if g.mean() < 12: k = 'black'
        elif cv2.Laplacian(g, cv2.CV_32F).var() < 1.5: k = 'occluded'          # thresholds are tunable
        elif p is not None and np.abs(g - p).mean() < .02:
            s.still += 1; k = 'frozen' if s.still > 20 else ''
        else: s.still = 0
        s.tp = k
        if k and s.ok(('tp', k), 60): s.ev.append(('tamper', 'critical', f'camera tamper: {k}'))

    def reid_job(s, t, cr):
        g, old = reid(cr, s.id); s.gid[t] = g
        if old: s.fire('reid', 'medium', f'person G{g} moved cam {old} -> cam {s.id}')
    def run_anpr_job(s, t, cr):
        try: s.anpr_job(t, cr)
        finally: s.anpr_pending.discard(t)
    def anpr_job(s, t, cr):
        reads = plate_reads(cr); consensus_read = None
        for raw, cf, img in reads:
            name = uuid.uuid4().hex + '.jpg'
            try: save_img(name, img)
            except Exception: name = ''
            q('insert into reads(ts,cam,track,raw,conf,crop) values(?,?,?,?,?,?)', (time.time(), s.id, t, raw, cf, name), True)      # every raw reading is kept
            if 5 <= len(raw) <= 12:
                if consensus_read is None or cf > consensus_read[1]: consensus_read = (raw, cf)
        if consensus_read:
            with PLT_LOCK:
                raw, cf = consensus_read; st = PLT.setdefault((s.id, t), {'r': [], 'v': None, 'id': None})
                if not st['r'] or raw != st['r'][-1][0]: st['r'].append((raw, max(.1, cf))); st['r'] = st['r'][-18:]
                v = vote(st['r'])
                if v:
                    first = st['id'] is None; st['v'] = v
                    if first: st['id'] = q('insert into plates(ts,cam,track,plate,votes) values(?,?,?,?,?)', (time.time(), s.id, t, v, len(st['r'])), True)
                    else: q('update plates set plate=?,votes=? where id=?', (v, len(st['r']), st['id']), True)
                    PLT[(s.id,t)]['v'] = v
                    if first: s.fire('plate', 'info', v)

def start(r):
    if r['id'] in CAMS: CAMS[r['id']].go = False
    c = Cam(r); CAMS[r['id']] = c; c.begin()

SIM_CAM_NEXT = 2000000000; SIM_CAM_LOCK = threading.Lock()
def create_sim_cam(name, src):
    global SIM_CAM_NEXT
    with SIM_CAM_LOCK:
        while SIM_CAM_NEXT in CAMS or q('select id from cams where id=?', (SIM_CAM_NEXT,)):
            SIM_CAM_NEXT += 1
        i = SIM_CAM_NEXT; SIM_CAM_NEXT += 1
    row = {'id':i,'name':name,'src':src,'fence':None,'fx':None,'fy':None,'simulation':True,'anpr_enabled':False}
    c = Cam(row); CAMS[i] = c; c.begin(); return c

def stop_sim_cams():
    for i,c in list(CAMS.items()):
        if c.simulation:
            c.go = False; CAMS.pop(i, None)

def sim_video_path(key):
    kind, sep, name = str(key or '').partition(':')
    if not sep or kind not in ('uploaded','demo') or not name or os.path.basename(name) != name or os.path.islink(os.path.join(D if kind == 'uploaded' else DEMO_VIDEO_DIR,name)): return None
    root = D + '/videos' if kind == 'uploaded' else DEMO_VIDEO_DIR
    path = os.path.abspath(os.path.join(root,name))
    return path if os.path.commonpath((os.path.abspath(root),path)) == os.path.abspath(root) and os.path.isfile(path) and os.path.splitext(name)[1].lower() in VIDEO_EXTS else None

def sim_video_list():
    videos=[]
    for kind,root in (('uploaded',D+'/videos'),('demo',DEMO_VIDEO_DIR)):
        try: names=sorted(os.listdir(root),key=str.casefold)
        except OSError: names=[]
        for name in names:
            path=sim_video_path(kind+':'+name)
            if path: videos.append(dict(id=kind+':'+name,name=name,source='Uploaded clip' if kind=='uploaded' else 'Preloaded clip'))
    return videos

SIM_SCANS = {}; SIM_SCAN_LOCK = threading.RLock()
def anpr_scan_worker(job_id, path):
    cap=None; model=None
    try:
        with SIM_SCAN_LOCK: job=SIM_SCANS[job_id]; job['status']='loading model'
        if OCR is None: raise RuntimeError('No OCR engine is available. Install RapidOCR or Tesseract, then restart IBVAP.')
        while not MODEL_READY.wait(.2):
            with SIM_SCAN_LOCK:
                if SIM_SCANS[job_id]['cancel'].is_set(): SIM_SCANS[job_id]['status']='cancelled'; return
        if MODEL_ERROR: raise RuntimeError('YOLO model is unavailable: '+str(MODEL_ERROR))
        model=YOLO(YOLO_PATH); cap=cv2.VideoCapture(path)
        if not cap.isOpened(): raise RuntimeError('The selected video could not be opened.')
        total=max(0,int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)); tracks={}; claimed=set(); next_tid=1; frame_no=0; started=time.time(); ocr_counts={}; votes={}; results={}; raw=[]
        with SIM_SCAN_LOCK: SIM_SCANS[job_id].update(status='scanning',total_frames=total)
        while True:
            with SIM_SCAN_LOCK:
                if SIM_SCANS[job_id]['cancel'].is_set(): SIM_SCANS[job_id]['status']='cancelled'; break
            ok,frame=cap.read()
            if not ok: break
            frame_no+=1
            if frame.shape[1] > 960: frame=cv2.resize(frame,(960,int(frame.shape[0]*960/frame.shape[1])),interpolation=cv2.INTER_AREA)
            out=model.track(frame,persist=True,tracker=TRACKER_PATH,classes=[2,3,5,7],conf=.20,imgsz=int(CFG['imgsz']),max_det=50,half=HALF,verbose=False,device=DEV)[0]
            claimed.clear(); now=time.time()
            for b in out.boxes:
                cls=int(b.cls[0]); x1,y1,x2,y2=map(int,b.xyxy[0].tolist()); raw_id=int(b.id[0]) if b.id is not None else -1
                tid,next_tid=assign_track_id((x1,y1,x2,y2),cls,raw_id,now,tracks,claimed,next_tid)
                sb,vx,vy=smooth_box((x1,y1,x2,y2),tracks.get(tid),now); tracks[tid]={'box':sb,'vx':vx,'vy':vy,'last':now,'cls':cls,'raw_id':raw_id}
                x1,y1,x2,y2=map(int,np.round(sb)); x1,y1=max(0,x1),max(0,y1); x2,y2=min(frame.shape[1],x2),min(frame.shape[0],y2); crop=frame[y1:y2,x1:x2]
                if crop.size == 0: continue
                ocr_counts[tid]=ocr_counts.get(tid,0)+1
                if ocr_counts[tid] % 6: continue
                try: candidates=plate_reads(crop)
                except Exception as e: print('simulation OCR:',repr(e)); continue
                if not candidates: continue
                plate,confidence,_=candidates[0]; votes.setdefault(tid,[]).append((plate,confidence)); votes[tid]=votes[tid][-18:]
                item=dict(frame=frame_no,track=tid,raw=plate,confidence=round(float(confidence),2)); raw.append(item); raw=raw[-200:]
                confirmed=vote(votes[tid])
                if confirmed: results[tid]=dict(track=tid,plate=confirmed,votes=len(votes[tid]),last_frame=frame_no)
            if frame_no % 5 == 0 or (total and frame_no >= total):
                with SIM_SCAN_LOCK:
                    job=SIM_SCANS[job_id]; job.update(processed_frames=frame_no,progress=round(frame_no*100/total,1) if total else 0,scan_fps=round(frame_no/max(.001,time.time()-started),1),raw_reads=list(raw),confirmed=list(results.values()))
        with SIM_SCAN_LOCK:
            if SIM_SCANS[job_id]['cancel'].is_set(): SIM_SCANS[job_id].update(status='cancelled',processed_frames=frame_no,scan_fps=round(frame_no/max(.001,time.time()-started),1),raw_reads=list(raw),confirmed=list(results.values()))
            elif SIM_SCANS[job_id]['status']!='cancelled': SIM_SCANS[job_id].update(status='complete',processed_frames=frame_no,progress=100,scan_fps=round(frame_no/max(.001,time.time()-started),1),raw_reads=list(raw),confirmed=list(results.values()))
    except Exception as e:
        with SIM_SCAN_LOCK:
            if job_id in SIM_SCANS: SIM_SCANS[job_id].update(status='error',error=str(e))
    finally:
        if cap: cap.release()
        del model

# ---------- API ----------
app = Flask(__name__)
@app.get('/')
def index(): return HTML

@app.post('/api/login')
def login():
    d = request.json or {}; username = str(d.get('u') or '').strip(); password = d.get('p')
    if not isinstance(password, str): password = ''
    r = q('select * from users where u=?', (username,))
    if r and hmac.compare_digest(ph(password, r[0]['h'].split('$')[0]), r[0]['h']): return jsonify(t=mk(r[0]['u'], r[0]['role']), r=r[0]['role'])
    return jsonify(error='bad credentials'), 401

@app.post('/api/password')
@auth()
def passwd():
    d = request.json or {}; r = q('select * from users where u=?', (request.u['u'],))[0]
    if not hmac.compare_digest(ph(d.get('old', ''), r['h'].split('$')[0]), r['h']): return jsonify(error='wrong current password')
    if len(d.get('new', '')) < 8: return jsonify(error='new password needs 8+ characters')
    q('update users set h=? where u=?', (ph(d['new']), r['u']), True); return jsonify(ok=1)

@app.route('/api/users', methods=['GET', 'POST'])
@auth(3)
def users():
    if request.method == 'POST':
        d = request.json or {}
        if d.get('role') not in RL or not d.get('u') or len(d.get('p', '')) < 8: return jsonify(error='username, role and a password of 8+ characters are required'), 400
        q('insert or replace into users values(?,?,?)', (d['u'], ph(d['p']), d['role']), True)
    return jsonify(q('select u,role from users'))

@app.get('/api/cams')
@auth()
def cams():
    out = []
    now = time.time()
    for r in q('select * from cams'):
        c = CAMS.get(r['id'])
        if c:
            with c.fence_lock: zones=[dict(id=zid,name=z['name']) for zid,z in c.fences.items()]
            with c.track_lock:
                track_snapshot = list(c.tracks.items()); seen_snapshot = list(c.track_seen.values())
            active = [(tid, st) for tid, st in track_snapshot if now - st['last'] <= 2.0]
            kinds = Counter('person' if st['cls'] == 0 else 'car' if st['cls'] in (2, 7) else 'motorcycle' if st['cls'] == 3 else 'bus' for _, st in active)
            tracked_10s = sum(1 for ts in seen_snapshot if now - ts <= 10)
        else: zones=q('select id,name from fences where cam=? order by id',(r['id'],))
        out.append(dict(r, fences=zones, status=c.status if c else 'off', ai_status=c.ai_state if c else 'off', ai_ready=bool(c.ai_ready) if c else False, cfps=round(c.cfps, 1) if c else 0, afps=round(c.afps, 1) if c else 0, ai_target=round(CFG['ai_fps'],1), ims=int(c.ims) if c else 0, W=c.W if c else 0, H=c.H if c else 0, tracks_live=len(active) if c else 0, tracks_10s=tracked_10s if c else 0, people=kinds.get('person',0) if c else 0, cars=kinds.get('car',0) if c else 0, motorcycles=kinds.get('motorcycle',0) if c else 0, buses=kinds.get('bus',0) if c else 0))
    return jsonify(out)

@app.get('/api/activity')
@auth()
def activity():
    end = time.time(); start = end - 12 * 3600; bins = [0] * 6
    for row in q('select cast((ts-?)/7200 as integer) bucket,count(*) n from alerts where ts>=? and ts<? group by bucket', (start, start, end)):
        bins[min(5, max(0, row['bucket']))] = row['n']
    recent = q('select count(*) n from alerts where ts>=?', (end - 86400,))[0]['n']
    day_key=datetime.datetime.fromtimestamp(end).date().isoformat(); day_start=datetime.datetime.fromtimestamp(end).replace(hour=0,minute=0,second=0,microsecond=0).timestamp()
    today_alerts=q('select count(*) n from alerts where ts>=? and ts<?',(day_start,end))[0]['n']
    today_incidents=q('select count(*) n from incidents where ts>=? and ts<?',(day_start,end))[0]['n']
    today_evidence=q("select count(*) n from alerts where ts>=? and ts<? and snap!=''",(day_start,end))[0]['n']
    today_plates=q('select count(*) n from plates where ts>=? and ts<?',(day_start,end))[0]['n']
    open_incidents=q("select count(*) n from incidents where status='open'")[0]['n']
    daily_tracks={'person':0,'car':0,'bike':0,'bus':0}
    for row in q('select kind,count(*) n from daily_tracks where day=? group by kind',(day_key,)):
        if row['kind'] in daily_tracks: daily_tracks[row['kind']]=row['n']
    return jsonify(start=start,end=end,bins=bins,alerts_24h=recent,day=day_key,daily_tracks=daily_tracks,daily_limits=read_daily_limits(),alerts_today=today_alerts,incidents_today=today_incidents,evidence_today=today_evidence,plates_today=today_plates,open_incidents=open_incidents)

@app.route('/api/daily-limits',methods=['GET','POST'])
@auth(2)
def daily_limits():
    if request.method=='POST':
        data=request.json or {}; limits={}
        for key in DAILY_LIMIT_KEYS:
            raw=data.get(key,0)
            try: value=float(raw)
            except (TypeError,ValueError): return jsonify(error=f'{key} daily cap must be a whole number'),400
            if isinstance(raw,bool) or not math.isfinite(value) or value!=int(value) or value<0 or value>1000000: return jsonify(error=f'{key} daily cap must be between 0 and 1,000,000'),400
            limits[key]=int(value)
        q("insert or replace into settings values('daily_limits',?)",(json.dumps(limits),),True)
        return jsonify(ok=1,limits=limits)
    return jsonify(limits=read_daily_limits())

@app.get('/api/cams/<int:i>/tracks')
@auth()
def cam_tracks(i):
    c = CAMS.get(i)
    if not c: return jsonify(error='unknown camera'), 404
    boxes = []
    for x1,y1,x2,y2,lab,col,vx,vy,ts in c.dets:
        color = 'red' if tuple(col) == RED else ('green' if tuple(col) == GREEN else 'blue')
        boxes.append(dict(x1=x1,y1=y1,x2=x2,y2=y2,label=lab,color=color,vx=vx,vy=vy,ts=ts))
    with c.fence_lock: fences=[dict(id=fid,name=z['name'],points=z['poly'].tolist()) for fid,z in c.fences.items()]
    return jsonify(ai_ready=c.ai_ready, ai_state=c.ai_state, status=c.status, cfps=round(c.cfps,1), afps=round(c.afps,1), ims=int(c.ims), ts=c.tlast, updated_at=c.det_updated_at, W=c.W, H=c.H, fence=fences[0]['points'] if fences else None, fences=fences, boxes=boxes)

@app.post('/api/cams')
@auth(3)
def cam_add():
    d = request.json or {}
    if not d.get('name') or not d.get('src'): return jsonify(error='name and source required'), 400
    src = d['src'].strip()
    if not (src.isdigit() or '://' in src or os.path.isfile(src) or (not os.path.isabs(src) and os.path.isfile(os.path.join(BASE, src)))): return jsonify(error='File not found on this computer'), 400
    if not src.isdigit() and '://' not in src and not os.path.isabs(src): src = os.path.abspath(os.path.join(BASE, src))
    i = q('insert into cams(name,src) values(?,?)', (d['name'], d['src'].strip()), True); start(q('select * from cams where id=?', (i,))[0]); return jsonify(id=i)

@app.post('/api/cams/upload')
@auth(3)
def cam_up():
    f = request.files['f']; p = f"{D}/videos/{uuid.uuid4().hex[:8]}_{re.sub(r'[^A-Za-z0-9._-]', '_', f.filename)}"; f.save(p)
    i = q('insert into cams(name,src) values(?,?)', (request.form.get('name') or f.filename, p), True); start(q('select * from cams where id=?', (i,))[0]); return jsonify(id=i)

@app.get('/api/simulation/videos')
@auth()
def sim_videos(): return jsonify(sim_video_list())

@app.post('/api/simulation/videos')
@auth(2)
def sim_video_upload():
    f=request.files.get('f')
    if not f or not f.filename: return jsonify(error='choose a video file first'),400
    original=os.path.basename(f.filename); ext=os.path.splitext(original)[1].lower()
    if ext not in VIDEO_EXTS: return jsonify(error='choose an MP4, AVI, MOV, MKV, WEBM, or M4V video'),400
    safe=re.sub(r'[^A-Za-z0-9._-]','_',original).strip('._') or 'demo'+ext
    path=os.path.join(D,'videos',uuid.uuid4().hex[:8]+'_'+safe); f.save(path)
    if os.path.getsize(path)>250*1024*1024: os.remove(path); return jsonify(error='simulation clips must be 250 MB or smaller'),413
    return jsonify(id='uploaded:'+os.path.basename(path),name=os.path.basename(path))

@app.get('/api/simulation/files/<kind>/<path:name>')
@auth()
def sim_video_file(kind,name):
    path=sim_video_path(kind+':'+name)
    return send_file(path,conditional=True) if path else (jsonify(error='video not found'),404)

@app.post('/api/simulation/webcam/start')
@auth(2)
def sim_webcam_start():
    stop_sim_cams(); c=create_sim_cam('Simulation Webcam','browser:'+uuid.uuid4().hex)
    return jsonify(id=c.id,name=c.r['name'],status='waiting for browser camera permission')

@app.post('/api/simulation/video/start')
@auth(2)
def sim_video_start():
    d=request.json or {}; path=sim_video_path(d.get('video'))
    if not path: return jsonify(error='select a preloaded or uploaded video'),400
    stop_sim_cams(); c=create_sim_cam('Simulation Video: '+os.path.basename(path),path)
    return jsonify(id=c.id,name=c.r['name'],status='loading')

@app.post('/api/simulation/camera/<int:i>/frame')
@auth(2)
def sim_webcam_frame(i):
    c=CAMS.get(i)
    if not c or not c.simulation or not c.r['src'].startswith('browser:'): return jsonify(error='webcam simulation is not active'),404
    item=request.files.get('frame')
    if not item: return jsonify(error='camera frame is missing'),400
    frame=cv2.imdecode(np.frombuffer(item.read(),np.uint8),cv2.IMREAD_COLOR)
    if frame is None: return jsonify(error='invalid camera frame'),400
    if frame.shape[1]>960: frame=cv2.resize(frame,(960,int(frame.shape[0]*960/frame.shape[1])),interpolation=cv2.INTER_AREA)
    now=time.time()
    if c.tlast: c.cfps=.9*c.cfps+.1/max(.001,now-c.tlast) if c.cfps else 1/max(.001,now-c.tlast)
    c.H,c.W=frame.shape[:2]; c.frame=frame; c.fid+=1; c.tlast=now; c.setst('online'); c.stream_event.set()
    return jsonify(ok=1)

@app.post('/api/simulation/camera/<int:i>/stop')
@auth(2)
def sim_camera_stop(i):
    c=CAMS.get(i)
    if not c or not c.simulation: return jsonify(ok=1)
    c.go=False; CAMS.pop(i,None); return jsonify(ok=1)

@app.post('/api/simulation/anpr/start')
@auth(2)
def sim_anpr_start():
    d=request.json or {}; path=sim_video_path(d.get('video'))
    if not path: return jsonify(error='select a preloaded or uploaded video'),400
    with SIM_SCAN_LOCK:
        if any(x['status'] in ('queued','loading model','scanning') and not x['cancel'].is_set() for x in SIM_SCANS.values()): return jsonify(error='an ANPR-only scan is already running'),409
        for key in list(SIM_SCANS):
            if SIM_SCANS[key]['status'] in ('complete','cancelled','error') and len(SIM_SCANS)>12: SIM_SCANS.pop(key,None)
        job_id=uuid.uuid4().hex; SIM_SCANS[job_id]={'id':job_id,'status':'queued','progress':0,'processed_frames':0,'total_frames':0,'scan_fps':0,'raw_reads':[],'confirmed':[],'error':'','cancel':threading.Event()}
    threading.Thread(target=anpr_scan_worker,args=(job_id,path),daemon=True).start()
    return jsonify(id=job_id)

@app.get('/api/simulation/anpr/<job_id>')
@auth(2)
def sim_anpr_status(job_id):
    with SIM_SCAN_LOCK:
        job=SIM_SCANS.get(job_id)
        if not job: return jsonify(error='scan not found'),404
        return jsonify({k:v for k,v in job.items() if k!='cancel'})

@app.post('/api/simulation/anpr/<job_id>/stop')
@auth(2)
def sim_anpr_stop(job_id):
    with SIM_SCAN_LOCK:
        job=SIM_SCANS.get(job_id)
        if not job: return jsonify(error='scan not found'),404
        job['cancel'].set()
    return jsonify(ok=1)

@app.delete('/api/cams/<int:i>')
@auth(3)
def cam_del(i):
    if i in CAMS: CAMS.pop(i).go = False
    q('delete from fences where cam=?',(i,),True); q('delete from cams where id=?', (i,), True); return jsonify(ok=1)

def fence_points(c,d):
    pts=d.get('pts',[])
    if len(pts)!=4: return None,'fence needs exactly 4 points'
    if not c.W or not c.H: return None,'camera has no video yet'
    try:
        sx=c.W/float(d.get('w') or c.W); sy=c.H/float(d.get('h') or c.H)
        p=[[int(max(0,min(c.W,float(x)*sx))),int(max(0,min(c.H,float(y)*sy)))] for x,y in pts]
    except (TypeError,ValueError,ZeroDivisionError): return None,'invalid fence coordinates'
    cx=sum(a[0] for a in p)/4; cy=sum(a[1] for a in p)/4
    p.sort(key=lambda a:math.atan2(a[1]-cy,a[0]-cx))
    if len(set(map(tuple,p)))<4: return None,'fence points must be distinct'
    return p,None

@app.post('/api/cams/<int:i>/fences')
@auth(2)
def fence_add(i):
    c=CAMS.get(i)
    if not c: return jsonify(error='unknown camera'),404
    d=request.json or {}; p,error=fence_points(c,d)
    if error: return jsonify(error=error),400
    with c.fence_lock:
        name=str(d.get('name') or '').strip()[:48] or f'Fence {len(c.fences)+1}'
        fid=q('insert into fences(cam,name,poly) values(?,?,?)',(i,name,json.dumps(p)),True)
        c.fences[fid]={'name':name,'poly':np.array(p,np.int32)}
    return jsonify(ok=1,id=fid,name=name)

@app.delete('/api/cams/<int:i>/fences/<int:fid>')
@auth(2)
def fence_remove(i,fid):
    c=CAMS.get(i)
    if not c: return jsonify(error='unknown fence'),404
    with c.fence_lock:
        if fid not in c.fences: return jsonify(error='unknown fence'),404
        c.fences.pop(fid); q('delete from fences where id=? and cam=?',(fid,i),True)
        for key in [k for k in c.din if k[0]==fid]: c.din.pop(key,None); c.loitered.discard(key)
    return jsonify(ok=1)

@app.delete('/api/cams/<int:i>/fences')
@auth(2)
def fences_clear(i):
    c=CAMS.get(i)
    if not c: return jsonify(error='unknown camera'),404
    with c.fence_lock:
        c.fences.clear(); c.din.clear(); c.loitered.clear(); q('delete from fences where cam=?',(i,),True)
    return jsonify(ok=1)

@app.route('/api/cams/<int:i>/fence', methods=['POST', 'DELETE'])
@auth(2)
def fence(i):
    c = CAMS.get(i)
    if not c: return jsonify(error='unknown camera'), 404
    if request.method == 'POST':
        p,error=fence_points(c,request.json or {})
        if error: return jsonify(error=error),400
        with c.fence_lock:
            if c.fences:
                fid=min(c.fences); name=c.fences[fid]['name']; q('update fences set poly=? where id=? and cam=?',(json.dumps(p),fid,i),True)
            else:
                name='Fence 1'; fid=q('insert into fences(cam,name,poly) values(?,?,?)',(i,name,json.dumps(p)),True)
            c.fences[fid]={'name':name,'poly':np.array(p,np.int32)}
    else:
        with c.fence_lock:
            if c.fences:
                fid=min(c.fences); c.fences.pop(fid); q('delete from fences where id=? and cam=?',(fid,i),True)
    return jsonify(ok=1)

@app.post('/api/cams/<int:i>/pos')
@auth(2)
def cam_pos(i):
    d = request.json or {}; q('update cams set fx=?,fy=? where id=?', (float(d['fx']), float(d['fy']), i), True); return jsonify(ok=1)

@app.get('/stream/<int:i>')
@auth()
def stream(i):
    def g():
        c = CAMS.get(i)
        if not c: return
        c.viewers += 1; last = -1
        try:
            while c.go:
                with c.stream_lock: jpg, n = c.jpg, c.n
                if jpg and n != last: last = n; yield b'--f\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n'
                else: time.sleep(.004)
        finally: c.viewers -= 1
    return Response(g(), mimetype='multipart/x-mixed-replace; boundary=f')

@app.get('/api/alerts')
@auth()
def alerts():
    w = "where a.snap!='' " if request.args.get('snap') else ''
    return jsonify(q(f'select a.*,c.name cname from alerts a left join cams c on c.id=a.cam {w}order by a.id desc limit 200'))

@app.get('/api/reads')
@auth()
def reads(): return jsonify(q('select r.*,c.name cname from reads r left join cams c on c.id=r.cam order by r.id desc limit 300'))

@app.get('/api/plates')
@auth()
def plates(): return jsonify(q('select p.*,c.name cname from plates p left join cams c on c.id=p.cam order by p.id desc limit 200'))

@app.post('/api/cams/<int:i>/snapshot')
@auth(2)
def cam_snap(i):
    c = CAMS.get(i)
    if not c or c.ai_frame is None: return jsonify(error='AI-processed video is not ready yet'), 400
    c.fire('manual_snapshot', 'info', 'Manual evidence snapshot by ' + request.u['u']); return jsonify(ok=1)

@app.get('/snap/<n>')
@auth()
def snap(n): return send_file(os.path.join(D, 'snaps', os.path.basename(n)), mimetype='image/jpeg')

@app.get('/api/incidents')
@auth()
def incs(): return jsonify(q('select * from incidents order by id desc limit 100'))

@app.get('/api/incidents/<int:i>')
@auth()
def inc_tl(i): return jsonify(q('select * from alerts where inc=? order by id', (i,)))

@app.post('/api/incidents/<int:i>/close')
@auth(2)
def inc_close(i): q("update incidents set status='closed' where id=?", (i,), True); return jsonify(ok=1)

@app.get('/api/incidents/<int:i>/export')
@auth(2)
def inc_export(i):
    a = q('select * from alerts where inc=? order by id', (i,)); b = io.BytesIO()
    with zipfile.ZipFile(b, 'w') as z:
        z.writestr('incident.json', json.dumps(dict(incident=q('select * from incidents where id=?', (i,)), alerts=a), indent=1))
        for x in a:
            if x['snap'] and os.path.exists(f"{D}/snaps/{x['snap']}"): z.write(f"{D}/snaps/{x['snap']}", x['snap'])
    b.seek(0); return send_file(b, as_attachment=True, download_name=f'incident_{i}.zip', mimetype='application/zip')

@app.route('/api/rules', methods=['GET', 'POST'])
@auth(3)
def rules():
    if request.method == 'POST':
        d = request.json or {}
        q('insert into rules(etype,cam,sev,incident,cnt,win) values(?,?,?,?,?,?)', (d.get('etype'), d.get('cam') or None, d.get('sev') or None, int(d.get('incident', 0)), max(1, int(d.get('cnt') or 1)), max(1, int(d.get('win') or 60))), True)
    return jsonify(q('select * from rules'))

@app.delete('/api/rules/<int:i>')
@auth(3)
def rule_del(i): q('delete from rules where id=?', (i,), True); return jsonify(ok=1)

@app.post('/api/floorplan')
@auth(3)
def fp_up():
    img = cv2.imdecode(np.frombuffer(request.files['f'].read(), np.uint8), 1)
    if img is None: return jsonify(error='bad image'), 400
    cv2.imwrite(D + '/floorplan.png', img); return jsonify(ok=1)

@app.get('/floorplan')
@auth()
def fp():
    p = D + '/floorplan.png'
    return send_file(p, mimetype='image/png') if os.path.exists(p) else ('', 404)

@app.route('/api/settings', methods=['GET', 'POST'])
@auth(3)
def settings():
    if request.method == 'POST':
        d = request.json or {}
        for k, lo, hi in (('ai_fps', 15, 120), ('dwell_s', 1, 3600), ('imgsz', 320, 960), ('stream_w', 320, 1920), ('mqtt_port', 1, 65535), ('anpr', 0, 1), ('reid', 0, 1)):
            if k in d: CFG[k] = type(CFG[k])(min(hi, max(lo, float(d[k]))))
        if 'mqtt_host' in d: CFG['mqtt_host'] = str(d['mqtt_host']).strip()
        q("insert or replace into settings values('cfg',?)", (json.dumps(CFG),), True); mq_connect()
    return jsonify(CFG)

@app.get('/api/verify')
@auth()
def verify():
    bad = []; p = Z; alerts=q('select * from alerts order by id'); ledger=q('select * from ledger order by id')
    for a in alerts:
        if a['prev'] != p or a['hash'] != hh(a['prev'], a['ts'], a['cam'], a['type'], a['sev'], a['info'], a['snap']): bad.append(a['id'])
        p = a['hash']
    p = Z
    for b in ledger:
        if b['prev'] != p or b['hash'] != sha(f"{p}|{b['ah']}|{b['ts']}"): bad.append(f"L{b['id']}")
        p = b['hash']
    if len(alerts)!=len(ledger): bad.append('alert/ledger entry count mismatch')
    for i,a in enumerate(alerts):
        if i>=len(ledger) or ledger[i]['ah']!=a['hash']: bad.append(f"alert #{a['id']} / ledger link")
    return jsonify(ok=not bad,bad=bad,alerts_checked=len(alerts),ledger_checked=len(ledger))

HTML = r"""<!doctype html><meta name=viewport content="width=device-width,initial-scale=1"><title>IBVAP | Intelligent Video Analytics Platform</title>
<style>
:root{--bg:#0a1220;--pn:#111d30;--pn2:#17263b;--ln:#293d58;--tx:#eaf2ff;--mu:#9aadc4;--ac:#41d7c4;--ac2:#7790ff;--ok:#48d597;--warn:#ffc76a;--bad:#ff7185}*{box-sizing:border-box}[hidden]{display:none!important}
body{margin:0;background:radial-gradient(ellipse at 78% 0%,#172a43 0,transparent 42%),var(--bg);color:var(--tx);font:14px/1.5 "Segoe UI",system-ui,sans-serif}
#app{display:flex;min-height:100vh}
aside{width:232px;flex:none;background:linear-gradient(180deg,#14243a,var(--pn) 45%);border-right:1px solid var(--ln);display:flex;flex-direction:column;position:sticky;top:0;height:100vh;padding:20px 12px}
.brand{font-size:18px;font-weight:750;letter-spacing:.7px;padding:0 10px 2px}.sub{color:var(--mu);font-size:11px;padding:0 10px 18px;letter-spacing:.4px;text-transform:uppercase}
nav{flex:1;overflow:auto}nav button{display:block;width:100%;text-align:left;background:none;border:0;border-left:3px solid transparent;border-radius:0 8px 8px 0;color:var(--mu);padding:10px 12px;margin:2px 0;cursor:pointer;font-size:14px;transition:background .15s,color .15s}
nav button:hover{color:var(--tx);background:#1a2b40}nav button.on{color:#f0fffd;background:#1a3548;border-left-color:var(--ac)}
#me{color:var(--mu);font-size:12px;padding:8px 10px}
main{flex:1;min-width:0;padding:28px}h2{margin:0 0 16px;font-size:23px;letter-spacing:-.25px}h3{margin:0 0 10px;font-size:15px}
.k{background:linear-gradient(145deg,rgba(21,36,57,.96),rgba(16,29,47,.98));border:1px solid var(--ln);border-radius:13px;padding:16px;overflow:auto;box-shadow:0 8px 26px #02091426}
.g{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px}
.st{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:16px}.st b{display:block;font-size:26px;margin-top:2px}
.th{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px}.th img{width:100%;border-radius:8px;display:block}
button,select,input{background:#182a40;color:var(--tx);border:1px solid #344c68;border-radius:8px;padding:8px 11px;margin:3px 0;font:inherit;transition:border-color .15s,background .15s}
input:focus,select:focus{outline:2px solid #41d7c455;border-color:var(--ac)}button{cursor:pointer}button:hover{border-color:#6b88a6;background:#203650}button.pri{background:linear-gradient(110deg,#22bdaa,#388dd8);border-color:#35c9bd;color:#061722;font-weight:700}
label{color:var(--mu);font-size:12px}table{width:100%;border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid var(--ln);text-align:left;vertical-align:middle}th{color:var(--mu);font-weight:600;font-size:12px}
.cap{text-transform:capitalize}.ok{color:var(--ok)}.bad,.critical,.high{color:var(--bad)}.medium{color:var(--warn)}small,.low,.info{color:var(--mu)}a{color:#76caff}
.v{position:relative;line-height:0;margin:8px 0}.v img,.v video{width:100%;border-radius:9px;background:#050a11;min-height:120px;display:block;object-fit:contain}.v canvas{position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none}
.vwait{position:absolute;inset:0;display:grid;place-items:center;background:#091321eF;color:var(--mu);border-radius:9px;line-height:1.4;text-align:center;padding:18px}.vwait[hidden]{display:none}
hr{border:0;border-top:1px solid var(--ln);margin:12px 0}.crop{height:26px;border-radius:3px;vertical-align:middle}
#toast{position:fixed;right:18px;top:18px;z-index:10;max-width:390px;padding:14px 18px;background:#192b3e;border:1px solid #41d7c4;border-radius:12px;box-shadow:0 12px 36px #0009;display:none;line-height:1.45}
.login-screen{max-width:none!important;margin:0!important;min-height:100vh;padding:30px clamp(20px,6vw,84px);display:flex;flex-direction:column;position:relative;overflow:hidden;background:radial-gradient(ellipse at 78% 45%,#18334a 0,transparent 40%)}
.login-brand{display:flex;align-items:center;gap:13px;position:relative;z-index:1}.login-brand strong{display:block;font-size:18px;letter-spacing:1.3px}.login-brand small{display:block;font-size:11px;letter-spacing:1.2px;text-transform:uppercase;color:var(--mu)}
.brandmark{width:42px;height:42px;display:grid;place-items:center;border:1px solid #41d7c477;border-radius:13px;background:linear-gradient(145deg,#1d514f,#203b64);box-shadow:0 0 28px #41d7c422}.brandmark svg{width:27px;height:27px}
:root{color-scheme:dark}
input,select,textarea,option{color-scheme:dark;background-color:#182a40;color:var(--tx)}
select option{background:#182a40;color:var(--tx)}input[type=file]::file-selector-button{background:#233852;color:var(--tx);border:1px solid #415a77;border-radius:6px;padding:6px 9px;margin-right:9px;cursor:pointer}
input::placeholder,textarea::placeholder{color:#8296ad;opacity:1}
input:-webkit-autofill,input:-webkit-autofill:hover,input:-webkit-autofill:focus{-webkit-text-fill-color:var(--tx);box-shadow:0 0 0 1000px #182a40 inset;transition:background-color 9999s ease-out}
input[type=checkbox]{accent-color:var(--ac);width:16px;height:16px;vertical-align:middle}
.login-stage{flex:1;display:grid;place-items:center;padding:28px 0 7vh}.login-card{width:min(100%,420px);padding:30px!important}.login-card h1{font-size:27px;margin:4px 0}.login-card p{color:var(--mu);margin:0 0 20px}.login-card label{display:block;margin:10px 0 2px}.login-card input{padding:11px 12px}.login-card .pri{width:100%;padding:11px;margin-top:14px}.login-kicker{font-size:10px;font-weight:700;letter-spacing:1.5px;color:var(--ac);text-transform:uppercase}
.password-field{display:flex;position:relative;width:100%;margin:0}.password-field input{width:100%;padding-right:76px!important}.password-toggle{position:absolute;right:7px;top:6px;bottom:6px;padding:0 12px;margin:0;border-color:transparent;background:#233852;color:var(--ac);font-size:12px;font-weight:700}.password-toggle:hover{background:#29415c;border-color:#3c5775}
#login-error{min-height:20px;margin-top:10px;color:var(--bad);font-size:12px;line-height:1.4}#login-error:empty{display:none}.login-hint{margin-top:8px;color:var(--mu);font-size:11px}.login-hint b{color:var(--tx)}
.login-submit:disabled{opacity:.7;cursor:wait}
.form-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:13px 14px}.field{display:grid;gap:5px;min-width:0}.field input,.field select{width:100%;min-width:0}.check-row{display:flex;flex-wrap:wrap;gap:8px 18px;margin:14px 0}.rule-list{display:grid;gap:8px;margin:12px 0}.rule-item{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:10px 12px;border:1px solid var(--ln);border-radius:9px;background:#101d30}.rule-summary{font-size:13px;line-height:1.4}.rule-builder{padding:14px;background:#0d1a2b;border:1px solid var(--ln);border-radius:10px}.rule-builder h4{margin:0 0 12px;color:var(--tx);font-size:14px}.rule-form-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.rule-form-grid .wide{grid-column:1/-1}
.dash-analytics{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(300px,1fr);gap:14px;margin:16px 0}.analytics-panel{background:linear-gradient(145deg,rgba(21,36,57,.96),rgba(16,29,47,.98));border:1px solid var(--ln);border-radius:13px;padding:16px;min-width:0}.analytics-head{display:flex;align-items:baseline;justify-content:space-between;gap:10px;margin-bottom:14px}.analytics-head h3{margin:0}.ring-grid{display:grid;grid-template-columns:repeat(4,minmax(100px,1fr));gap:10px}.ring-card{display:grid;justify-items:center;text-align:center;gap:8px;padding:8px 4px}.ring-meter{width:90px;height:90px;border-radius:50%;display:grid;place-items:center;background:conic-gradient(var(--ring) calc(var(--pct)*1%),#293d58 0);position:relative}.ring-meter:before{content:'';position:absolute;inset:8px;border-radius:50%;background:#111d30}.ring-meter b{position:relative;z-index:1;font-size:22px}.ring-name{font-size:13px;font-weight:700}.ring-meta{font-size:10px;color:var(--mu);line-height:1.35}.alert-bars{height:148px;display:grid;grid-template-columns:repeat(6,minmax(0,1fr));align-items:end;gap:10px;padding:8px 4px 0}.alert-bar-col{height:100%;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;gap:6px;min-width:0}.alert-bar-value{font-size:11px;color:var(--tx);min-height:14px}.alert-bar{width:min(30px,60%);min-height:3px;border-radius:7px 7px 2px 2px;background:linear-gradient(180deg,var(--ac),#4387d1);box-shadow:0 0 16px #41d7c422}.alert-bar-label{font-size:10px;color:var(--mu)}
.track-summary{display:flex;align-items:center;gap:22px;flex-wrap:wrap;margin-top:10px;padding:11px 12px;border:1px solid var(--ln);border-radius:9px;background:#0d1a2b}.track-summary div{display:grid;gap:2px}.track-summary b{font-size:20px}.metric-note,.chart-empty{font-size:11px;color:var(--mu)}.chart-empty{display:block;text-align:center;margin-top:2px}
.field input[type=checkbox]{width:16px;height:16px}.field label,.field{color:#b5c6d9}.field input,.field select{background:#182a40;border-color:#344c68}.field input:focus,.field select:focus{outline:2px solid #41d7c455;border-color:var(--ac)}.check-field>span:last-child{display:flex;align-items:center;gap:8px;color:var(--mu)}.check-field>span:last-child input{flex:none}
.simulation-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.sim-card{background:linear-gradient(145deg,#152439f5,#101d2ffe);border:1px solid var(--ln);border-radius:13px;padding:16px;min-width:0;box-shadow:0 8px 26px #02091426}.sim-card h3{margin-bottom:5px}.sim-card>p{margin:0 0 12px;color:var(--mu);font-size:12px}.sim-card.wide{grid-column:1/-1}.sim-actions{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}.sim-status{display:inline-flex;align-items:center;gap:7px;border:1px solid var(--ln);border-radius:99px;padding:5px 10px;font-size:11px;font-weight:700;background:#101b2b}.sim-status:before{content:'';width:7px;height:7px;border-radius:50%;background:currentColor}.sim-normal{color:var(--ok)}.sim-warning{color:var(--warn)}.sim-danger{color:var(--bad)}.sim-neutral{color:var(--mu)}.sim-feed{display:grid;gap:7px;margin-top:12px;max-height:150px;overflow:auto}.sim-event{display:flex;justify-content:space-between;gap:12px;padding:8px 10px;border:1px solid var(--ln);border-radius:8px;background:#0d1a2b;font-size:12px}.sim-video-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(200px,.8fr);align-items:start;gap:12px;margin-top:12px}.sim-progress{height:8px;border-radius:8px;background:#293d58;overflow:hidden;margin:10px 0}.sim-progress>i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--ac),#638bff);transition:width .2s}.sim-result-table{max-height:280px;overflow:auto}.sim-result-table table{font-size:12px}.sim-result-table td,.sim-result-table th{padding:6px}.sim-file-row{display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:end;gap:8px}.sim-help{font-size:11px;color:var(--mu)}
.sim-health-screen{min-height:118px;display:grid;align-content:center;justify-items:center;gap:4px;margin-bottom:10px;border:1px solid var(--ln);border-radius:10px;background:radial-gradient(ellipse at 50% 40%,#1b4250,#0a1422 75%);color:#a8f0e6;text-align:center}.sim-health-screen.tamper{background:repeating-linear-gradient(0deg,#151a22 0 4px,#20232a 4px 8px);color:#ffd083}.sim-health-screen.loss{background:#05080d;color:#ff8292}.sim-health-screen small{font-size:9px;letter-spacing:1.5px}.sim-health-screen strong{font-size:16px;letter-spacing:1px}.sim-health-screen span{font-size:10px;color:var(--mu)}
@media(max-width:760px){#app{flex-direction:column}aside{width:auto;height:auto;position:static}main{padding:18px}nav{display:flex;flex-wrap:wrap}nav button{width:auto}.login-screen{padding:22px}.login-stage{padding-bottom:3vh}}
@media(max-width:980px){.dash-analytics{grid-template-columns:1fr}.ring-grid{grid-template-columns:repeat(4,minmax(90px,1fr))}}
@media(max-width:800px){.simulation-grid{grid-template-columns:1fr}.sim-card.wide{grid-column:auto}.sim-video-grid{grid-template-columns:1fr}}
@media(max-width:520px){.form-grid,.rule-form-grid{grid-template-columns:1fr}.rule-form-grid .wide{grid-column:auto}.ring-grid{grid-template-columns:repeat(2,minmax(100px,1fr))}.analytics-head{align-items:flex-start;flex-direction:column}.sim-file-row{grid-template-columns:1fr}}
</style>
<div id=lg class=login-screen><header class=login-brand><span class=brandmark aria-hidden=true><svg viewBox="0 0 32 32" fill="none"><path d="M5 24V17h5v7M13.5 24V11h5v13M22 24V5h5v19" stroke="#7ff2df" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/><path d="M4 27.5h24" stroke="#83a5ff" stroke-width="2" stroke-linecap="round"/></svg></span><span><strong>IBVAP</strong><small>Intelligent Video Analytics Platform</small></span></header><div class=login-stage><div class="k login-card"><div class=login-kicker>Secure access</div><h1>Welcome back</h1><p>Sign in to your analytics command center.</p><label for=u>Username</label><input id=u autocomplete=username style=width:100%><label for=p>Password</label><div class=password-field><input id=p type=password autocomplete=current-password onkeydown="if(event.key=='Enter')login()"><button type=button class=password-toggle onclick="togglePassword()" aria-pressed=false>Show</button></div><div id=login-error role=alert aria-live=polite></div><div class=login-hint>New install: <b>admin</b> / <b>admin123</b> (no space). Existing installs keep their saved password.</div><button class="pri login-submit" onclick=login()>Sign In</button></div></div></div>
<div id=toast role=alert></div><div id=app hidden><aside><div class=brand>IBVAP</div><div class=sub>Intelligent Video Analytics</div><nav id=nv></nav><div id=me></div><button onclick="localStorage.clear();location.reload()">Sign Out</button></aside><main id=m></main></div>
<script>
let T=localStorage.t,R=localStorage.r,tab='dashboard',F=null,lastAlert=0,TRACK_IDS=[],TRACKS={};const SIM={health:'normal',healthEvents:[],webcamCam:null,webcamStream:null,webcamRaf:0,webcamBusy:false,videoCam:null,scanId:null,scanTimer:0,videos:[]};const $=s=>document.querySelector(s);
const E=s=>String(s??'').replace(/[&<>"']/g,c=>'&#'+c.charCodeAt(0)+';'),O=(a,f)=>a.map(f).join(''),tm=t=>new Date(t*1e3).toLocaleTimeString(),IM=n=>`/snap/${n}?t=${T}`;
async function api(u,o={}){o.headers={Authorization:'Bearer '+T,...(o.headers||{})};
 if(o.body&&!(o.body instanceof FormData)){o.body=JSON.stringify(o.body);o.headers['Content-Type']='application/json'}
 const r=await fetch(u,o);if(r.status==401){localStorage.clear();location.reload()}
 return (r.headers.get('content-type')||'').includes('json')?r.json():r}
function togglePassword(){const p=$('#p'),b=$('.password-toggle'),show=p.type==='password';p.type=show?'text':'password';b.textContent=show?'Hide':'Show';b.setAttribute('aria-pressed',String(show));p.focus()}
async function login(){const b=$('.login-submit'),m=$('#login-error');m.textContent='';b.disabled=true;b.textContent='Signing in…';try{const r=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({u:$('#u').value.trim(),p:$('#p').value})});const d=await r.json();if(d.t){localStorage.t=d.t;localStorage.r=d.r;location.reload()}else{m.textContent='Login failed. Use admin123 with no space on a fresh install. Existing installs keep the password previously saved. To reset it, close the app and run reset_admin.bat.'}}catch(e){m.textContent='Cannot reach the local app service. Close and reopen the desktop app, then try again.'}finally{b.disabled=false;b.textContent='Sign In'}}
const NAV=[['dashboard','Dashboard'],['live','Live View'],['source','Add Source'],['alerts','Alerts'],['anpr','ANPR Readings'],['incidents','Incidents'],['evidence','Evidence'],['map','Floor Plan']].concat(R!='viewer'?[['limits','Daily Limits'],['simulation','Simulation']]:[],R=='admin'?[['rules','Rules & Settings'],['admin','Users & Integrity']]:[],[['account','Account']]);
async function init(){if(!T)return;$('#lg').hidden=1;$('#app').hidden=0;$('#nv').innerHTML=O(NAV,n=>`<button id=b${n[0]} onclick="go('${n[0]}')">${n[1]}</button>`);$('#me').textContent='Signed In As '+R;const a=await api('/api/alerts');lastAlert=a.length?a[0].id:0;go('dashboard')}
function go(t){if(tab==='simulation'&&t!=='simulation')closeSimulation();tab=t;F=null;NAV.forEach(n=>$('#b'+n[0]).classList.toggle('on',n[0]==t));draw()}
function draw(){({dashboard,live,source,alerts,anpr,incidents,evidence,map:plan,limits:dailyLimits,rules,admin,account,simulation})[tab]()}
setInterval(()=>{if(T)({dashboard,live:stats,alerts,anpr})[tab]?.()},3000);
async function pollLoiter(){if(!T)return;try{const a=await api('/api/alerts');for(const x of a.slice().reverse())if(x.id>lastAlert){lastAlert=x.id;if(x.type==='loitering'){$('#toast').innerHTML='<b>Loitering Alert</b><br>'+E(x.info)+(x.snap?`<br><a href="${IM(x.snap)}" target=_blank>View evidence snapshot</a>`:'');$('#toast').style.display='block';setTimeout(()=>$('#toast').style.display='none',9000)}}if(a.length)lastAlert=Math.max(lastAlert,a[0].id)}catch(e){}}
setInterval(pollLoiter,2500);
const al=x=>`<tr><td>${tm(x.ts)}</td><td>${E(x.cname||x.cam)}</td><td class="cap ${x.sev}">${x.type.replace(/_/g,' ')}</td><td>${E(x.info)}</td><td>${x.snap?`<a target=_blank href="${IM(x.snap)}">View</a>`:''}</td></tr>`;
async function dashboard(){const [c,a,activity]=await Promise.all([api('/api/cams'),api('/api/alerts'),api('/api/activity')]),ev=a.filter(x=>x.snap);
const totals=activity?.daily_tracks||{person:0,car:0,bike:0,bus:0},caps=activity?.daily_limits||{},dailyTotal=Object.values(totals).reduce((n,x)=>n+x,0);
const colors={person:'#41d7c4',car:'#7790ff',bike:'#ffc76a',bus:'#ff7185'},labels={person:'People',car:'Cars',bike:'Bikes',bus:'Buses'},ringHtml=Object.entries(totals).map(([name,count])=>{const cap=Number(caps[name]||0),pct=cap?Math.min(100,Math.round(count*100/cap)):0,over=cap>0&&count>cap,color=over?'#ff7185':colors[name],meta=cap?`${count} / ${cap} · ${over?'limit exceeded':pct+'%'}`:'No daily cap set';return `<div class=ring-card><div class=ring-meter style="--ring:${color};--pct:${pct}" role="img" aria-label="${count} of ${cap||'no'} daily ${labels[name].toLowerCase()} limit"><b>${count}</b></div><div class=ring-name>${labels[name]}</div><div class=ring-meta>${meta}</div></div>`}).join('');
const bins=activity?.bins||[0,0,0,0,0,0],maxBin=Math.max(1,...bins),barHtml=bins.map((n,k)=>`<div class=alert-bar-col><span class=alert-bar-value>${n}</span><div class=alert-bar style="height:${n?Math.max(7,Math.round(n/maxBin*100)):3}px"></div><span class=alert-bar-label>${['10h','8h','6h','4h','2h','Now'][k]}</span></div>`).join('');
$('#m').innerHTML=`<h2>Dashboard</h2><div class=st><div class=k><small>Sources Online</small><b>${c.filter(x=>x.status=='online').length} / ${c.length}</b></div>
<div class=k><small>Alerts Today</small><b>${activity?.alerts_today||0}</b></div><div class=k><small>Incidents Today</small><b>${activity?.incidents_today||0}</b></div>
<div class=k><small>Open Incidents</small><b>${activity?.open_incidents||0}</b></div><div class=k><small>Evidence Today</small><b>${activity?.evidence_today||0}</b></div><div class=k><small>Plates Confirmed Today</small><b>${activity?.plates_today||0}</b></div></div>
<div class=dash-analytics><section class=analytics-panel><div class=analytics-head><h3>Today's detected objects</h3><small>Distinct track IDs recorded since local midnight</small></div><div class=ring-grid>${ringHtml}</div><div class=track-summary><div><small>Tracks recorded today</small><b>${dailyTotal}</b></div><small class=metric-note>Daily counts are stored per camera and tracker session; the same person or vehicle can count again when it receives a new track.</small></div></section>
<section class=analytics-panel><div class=analytics-head><h3>Alert activity</h3><small>Two-hour windows over the last 12 hours</small></div><div class=alert-bars role=img aria-label="Alert activity bar chart for the last 12 hours">${barHtml}</div>${bins.every(n=>n===0)?'<small class=chart-empty>No alerts recorded in this period.</small>':''}</section></div>
<div class=g><div class=k><h3>Sources</h3><table>${O(c,x=>`<tr><td>${E(x.name)}</td><td class="cap ${x.status=='online'?'ok':'bad'}">${x.status}</td><td>${x.cfps} FPS</td><td>AI ${x.afps} FPS</td></tr>`)||'<tr><td><small>No Sources Yet</small></td></tr>'}</table></div>
<div class=k><h3>Recent Alerts</h3><table>${O(a.slice(0,8),al)||'<tr><td><small>No Alerts Yet</small></td></tr>'}</table></div></div>
<h3 style="margin-top:18px">Latest Evidence</h3><div class=th>${O(ev.slice(0,6),x=>`<a target=_blank href="${IM(x.snap)}"><img src="${IM(x.snap)}"></a>`)||'<small>No Evidence Yet</small>'}</div>`}
async function dailyLimits(){const r=await api('/api/daily-limits'),d=r.limits||{person:0,car:0,bike:0,bus:0};$('#m').innerHTML=`<h2>Daily Limits</h2><p class=sim-help>Set the daily count target for each object category. The Dashboard rings compare today's tracked totals with these values.</p><section class=k style="max-width:780px"><h3>Daily object caps</h3><div class=form-grid><label class=field>People per day<input id=dl-person type=number min=0 max=1000000 step=1 value="${d.person||''}" placeholder="No cap"></label><label class=field>Cars per day<input id=dl-car type=number min=0 max=1000000 step=1 value="${d.car||''}" placeholder="No cap"></label><label class=field>Bikes per day<input id=dl-bike type=number min=0 max=1000000 step=1 value="${d.bike||''}" placeholder="No cap"></label><label class=field>Buses per day<input id=dl-bus type=number min=0 max=1000000 step=1 value="${d.bus||''}" placeholder="No cap"></label></div><button class=pri onclick="saveDailyLimits()">Save Daily Limits</button><div id=daily-limit-status class=metric-note aria-live=polite></div><p><small>Leave a field blank or enter 0 for no cap. These targets only affect the dashboard rings; they do not block entry or create alerts. Totals count distinct track IDs per camera and tracker session, not unique real-world identities.</small></p></section>`}
async function saveDailyLimits(){const status=$('#daily-limit-status'),values={};for(const key of ['person','car','bike','bus']){const raw=$('#dl-'+key).value,value=raw===''?0:Number(raw);if(!Number.isSafeInteger(value)||value<0||value>1000000){status.textContent='Enter whole-number limits from 0 to 1,000,000.';return}values[key]=value}const r=await api('/api/daily-limits',{method:'POST',body:values});status.textContent=r.error||'Daily limits saved. The dashboard will use them the next time it refreshes.';if(r.ok)status.className='ok'}
async function live(){const c=await api('/api/cams');TRACK_IDS=c.map(x=>x.id);$('#m').innerHTML='<h2>Live View</h2><p id=hint><small>Original source video. Boxes show the latest smoothed AI tracks; Yellow = Fence, Red = Inside Fence, Green = Person, Blue = Vehicle.</small></p>'+(c.length?'':'<div class=k>No Sources Yet. Open Add Source To Connect A Camera Or Video.</div>')+'<div class=g>'+ 
 O(c,x=>`<div class=k><b>${E(x.name)}</b><br><small id=st${x.id}></small><div class=v><img id=im${x.id} onload="$('#wait${x.id}').hidden=true" src="/stream/${x.id}?t=${T}"><div class=vwait id=wait${x.id}>${x.ai_ready?'Waiting for the first analyzed frame…':'AI is '+E(x.ai_status||'loading')+'…'}</div><canvas id=cv${x.id}></canvas></div><button onclick="fence(${x.id})">Add 4-Point Fence</button> <button onclick="clr(${x.id})">Clear All Fences</button> <button onclick="snap(${x.id})">Save Snapshot</button><div id=fl${x.id}><small>${O(x.fences||[],z=>`${E(z.name)} <button onclick="removeFence(${x.id},${z.id})">Remove</button>`)}</small></div></div>`)+'</div>';stats(c)}
async function stats(c){c=c||await api('/api/cams');c.forEach(x=>{const e=$('#st'+x.id),w=$('#wait'+x.id),target=x.ai_target||30;if(e)e.textContent=`${x.status.toUpperCase()} | ${x.ai_ready?'AI READY':'AI '+String(x.ai_status||'loading').toUpperCase()} | Capture ${x.cfps} FPS | AI ${x.afps} / ${target} FPS${x.afps<target?(x.cfps<target?' (SOURCE BELOW TARGET)':' (UNDER TARGET)'):''} | Inference ${x.ims} ms | ${x.W}x${x.H}`;if(w&&!x.ai_ready)w.textContent=x.ai_status==='error'?'AI could not start. Check the server message.':'AI is '+String(x.ai_status||'loading')+'; waiting for the video…'})}
let trackPollBusy=false;
async function pullTracks(){if(!T||!['live','simulation'].includes(tab)||trackPollBusy)return;trackPollBusy=true;const ids=tab==='live'?TRACK_IDS:[SIM.webcamCam,SIM.videoCam].filter(Boolean);try{await Promise.all(ids.map(async i=>{try{TRACKS[i]=await api(`/api/cams/${i}/tracks`);const el=$('#'+(i===SIM.webcamCam?'sim-webcam-info':'sim-video-info')),x=TRACKS[i];if(el)el.textContent=`${String(x.status||'loading').toUpperCase()} | Capture ${x.cfps} FPS | AI ${x.afps} FPS | ${x.ims} ms inference`}catch(e){}}))}finally{trackPollBusy=false}}
setInterval(pullTracks,50);
function drawTracks(){const ids=tab==='live'?TRACK_IDS:tab==='simulation'?[SIM.webcamCam,SIM.videoCam].filter(Boolean):[];for(const id of ids){const cv=$('#cv'+id),im=$('#im'+id),tr=TRACKS[id];if(!cv||!im||!im.clientWidth||!im.clientHeight)continue;const w=Math.round(im.clientWidth),h=Math.round(im.clientHeight);if(cv.width!==w)cv.width=w;if(cv.height!==h)cv.height=h;const g=cv.getContext('2d');g.clearRect(0,0,w,h);if(!tr||!tr.W||!tr.H)continue;const sx=w/tr.W,sy=h/tr.H,boxDataAge=Math.max(0,Date.now()/1000-(tr.updated_at||tr.ts||0));
 (tr.fences|| (tr.fence?[{name:'Fence',points:tr.fence}]:[])).forEach((z,k)=>{g.strokeStyle=['#ffeb3b','#ff9f43','#40c9ff','#ff6bcb','#a0e75a'][k%5];g.lineWidth=2;g.beginPath();z.points.forEach((p,j)=>j?g.lineTo(p[0]*sx,p[1]*sy):g.moveTo(p[0]*sx,p[1]*sy));g.closePath();g.stroke();g.font='bold 12px Segoe UI, sans-serif';g.fillText(z.name,z.points[0][0]*sx,z.points[0][1]*sy-4)})
  if(boxDataAge<=2.0)for(const b of tr.boxes||[]){const age=Date.now()/1000-b.ts,dt=Math.max(0,Math.min(.2,age)),x1=Math.max(0,b.x1+b.vx*dt),x2=Math.min(tr.W,b.x2+b.vx*dt),y1=Math.max(0,b.y1+b.vy*dt),y2=Math.min(tr.H,b.y2+b.vy*dt);g.strokeStyle=b.color==='red'?'#ff3030':b.color==='green'?'#20df65':'#3287ff';g.lineWidth=2;g.strokeRect(x1*sx,y1*sy,(x2-x1)*sx,(y2-y1)*sy);g.font='bold 14px Segoe UI, sans-serif';g.fillStyle=g.strokeStyle;g.fillText(b.label,x1*sx,Math.max(14,y1*sy-5))}
 if(F&&F.id===id&&F.p.length){g.strokeStyle=g.fillStyle='#ff0';g.lineWidth=2;g.beginPath();F.p.forEach((p,k)=>k?g.lineTo(p[0],p[1]):g.moveTo(p[0],p[1]));if(F.p.length===4)g.closePath();g.stroke();F.p.forEach(p=>g.fillRect(p[0]-3,p[1]-3,6,6))}}
 requestAnimationFrame(drawTracks)}
requestAnimationFrame(drawTracks);
function fence(i){const name=prompt('Name this fence',`Fence ${(TRACKS[i]?.fences||[]).length+1}`);if(name===null)return;F={id:i,p:[],name:name.trim()||`Fence ${(TRACKS[i]?.fences||[]).length+1}`};const cv=$('#cv'+i),im=$('#im'+i);cv.width=im.clientWidth;cv.height=im.clientHeight;cv.style.pointerEvents='auto';cv.style.cursor='crosshair';
 cv.onclick=e=>{F.p.push([e.offsetX,e.offsetY]);sk(cv)};$('#hint').textContent='Click four corners to add this fence. Existing fences stay in place.'}
function sk(cv){drawTracks();if(F.p.length==4)save(cv)}
async function save(cv){const i=F.id,name=F.name,im=$('#im'+i),sx=im.naturalWidth/cv.width,sy=im.naturalHeight/cv.height,pts=F.p.map(p=>[p[0]*sx,p[1]*sy]);
 const r=await api(`/api/cams/${i}/fences`,{method:'POST',body:{name,pts,w:im.naturalWidth,h:im.naturalHeight}});cv.onclick=null;cv.style.pointerEvents='none';F=null;
 if(r.error){$('#hint').textContent=r.error;return}await pullTracks();renderFenceList(i);drawTracks();$('#hint').textContent=`${r.name} saved. You can add more fences.`}
function renderFenceList(i){const el=$('#fl'+i),zs=TRACKS[i]?.fences||[];if(el)el.innerHTML=`<small>${O(zs,z=>`${E(z.name)} <button onclick="removeFence(${i},${z.id})">Remove</button>`)}</small>`}
async function removeFence(i,fid){const r=await api(`/api/cams/${i}/fences/${fid}`,{method:'DELETE'});if(r.error){$('#hint').textContent=r.error;return}await pullTracks();renderFenceList(i);drawTracks()}
async function clr(i){const r=await api(`/api/cams/${i}/fences`,{method:'DELETE'});if(r.error)$('#hint').textContent=r.error;else{await pullTracks();renderFenceList(i);drawTracks();$('#hint').textContent='All fences cleared'}}
async function snap(i){const r=await api(`/api/cams/${i}/snapshot`,{method:'POST'});$('#hint').textContent=r.error||'Snapshot Saved To Evidence'}
async function source(){const c=await api('/api/cams');$('#m').innerHTML=`<h2>Add Source</h2><div class=g><section class=k><h3>Connect a camera or video</h3><div class=form-grid>
<label class=field>Source type<select id=ty onchange="$('#fc').hidden=this.value!='cam';$('#fv').hidden=this.value!='vid'"><option value=cam>IP camera or webcam</option><option value=vid>Video file</option></select></label>
<label class=field>Source name<input id=sn placeholder="Front Gate"></label>
<div id=fc class="field wide"><label for=sa>Camera address</label><input id=sa placeholder="RTSP/HTTP URL or 0 for webcam"></div>
<div id=fv class="field wide" hidden><label for=sf>Choose video file<input type=file id=sf accept="video/*"></label><label for=sp>Or enter a file path</label><input id=sp placeholder="C:/Videos/clip.mp4"></div></div>
<button class=pri onclick="addsrc()">Add source</button><div id=msg class=bad role=status></div></section>
<section class=k><h3>Configured sources</h3><table>${O(c,x=>`<tr><td>${E(x.name)}<br><small>${E(x.src)}</small></td><td class="cap ${x.status=='online'?'ok':'bad'}">${x.status}</td><td><button onclick="del('cams',${x.id})">Remove</button></td></tr>`)||'<tr><td><small>No sources yet</small></td></tr>'}</table></section></div>`}
async function addsrc(){const t=$('#ty').value,n=$('#sn').value.trim(),m=$('#msg');let r;if(!n)return m.textContent='Enter A Source Name';
 if(t=='cam'){const s=$('#sa').value.trim();if(!s)return m.textContent='Enter The Camera Address';r=await api('/api/cams',{method:'POST',body:{name:n,src:s}})}
 else{const f=$('#sf').files[0],p=$('#sp').value.trim();if(f){const d=new FormData();d.append('name',n);d.append('f',f);m.textContent='Uploading...';r=await api('/api/cams/upload',{method:'POST',body:d})}
  else if(p)r=await api('/api/cams',{method:'POST',body:{name:n,src:p}});else return m.textContent='Choose A Video File'}
 if(r.error)m.textContent=r.error;else go('live')}
async function simulation(){const v=await api('/api/simulation/videos');SIM.videos=Array.isArray(v)?v:[];$('#m').innerHTML=`<h2>Simulation</h2><p class=sim-help>Run isolated demos for camera health, browser webcam tracking, video tracking, or ANPR-only scans. Demo activity is kept separate from operational alerts and evidence.</p><div class=simulation-grid>
<section class="sim-card wide"><h3>Camera health scenarios</h3><p>Demonstrate how operators identify tampering and loss of video. These controls simulate a health event in this panel only; they do not alter a connected camera or write to the operational alert log.</p><div id=sim-health-status></div><div class=sim-actions><button onclick="simulateHealth('tamper')">Simulate camera tampering</button><button onclick="simulateHealth('loss')">Simulate detection loss</button><button onclick="simulateHealth('normal')">Restore normal status</button></div><div id=sim-health-feed class=sim-feed></div></section>
<section class=sim-card><h3>Browser webcam tracking</h3><p>Ask the browser for camera permission, show the original webcam feed, and send resized frames to the local YOLO tracker.</p><div class=sim-actions><label>Frame target <select id=sim-webcam-rate><option>15</option><option selected>20</option><option>30</option></select> FPS</label><button class=pri onclick="startWebcamSim()">Request webcam access</button><button onclick="stopWebcamSim()">Stop webcam</button></div><div id=sim-webcam-status class=sim-help>Camera permission has not been requested.</div><div id=sim-webcam-info class=sim-help></div><div class="v" id=sim-webcam-preview hidden><video id=sim-webcam-video autoplay muted playsinline></video></div></section>
<section class=sim-card><h3>Video tracking simulation</h3><p>Play a local test clip through the normal detector and smoothed tracker. This temporary simulation camera does not become a saved CCTV source.</p><label class=field>Preloaded or previously uploaded clip<select id=sim-video-choice></select></label><div class=sim-actions><button class=pri onclick="startVideoSim()">Run video tracking</button><button onclick="stopVideoSim()">Stop video</button></div><div id=sim-video-info class=sim-help></div><div class=v id=sim-video-preview hidden></div></section>
<section class="sim-card wide"><h3>ANPR-only video scan</h3><p>Process one selected clip for vehicle plates only. Raw OCR candidates and confidence-weighted confirmed readings appear here; this scan does not add person tracking or operational camera alerts.</p><div class=sim-file-row><label class=field>Clip for ANPR scan<select id=sim-anpr-choice></select></label><div class=sim-actions><button class=pri onclick="startAnprSim()">Scan for plates</button><button onclick="stopAnprSim()">Cancel scan</button></div></div>
<div class=sim-file-row style="margin-top:8px"><label class=field>Add a local video to the simulation library<input id=sim-upload type=file accept=".mp4,.avi,.mov,.mkv,.webm,.m4v,video/*"></label><button onclick="uploadSimVideo()">Add video</button></div><div id=sim-library-message class=sim-help>Preloaded clips in demo_videos and clips already uploaded in Add Source appear in both selectors.</div>
<div id=sim-anpr-player class=v hidden></div><div id=sim-anpr-status class=sim-help></div><div class=sim-progress><i id=sim-anpr-progress></i></div><div class=sim-video-grid><div class=sim-result-table><h4>Confirmed plates</h4><table><thead><tr><th>Track</th><th>Plate</th><th>Votes</th><th>Last frame</th></tr></thead><tbody id=sim-anpr-confirmed><tr><td colspan=4 class=sim-help>No scan results yet.</td></tr></tbody></table></div><div class=sim-result-table><h4>Raw OCR reads</h4><table><thead><tr><th>Frame</th><th>Track</th><th>Reading</th><th>Conf.</th></tr></thead><tbody id=sim-anpr-raw><tr><td colspan=4 class=sim-help>No scan results yet.</td></tr></tbody></table></div></div></section></div>`;
renderHealthSim();await refreshSimVideos();if(SIM.webcamStream){$('#sim-webcam-preview').hidden=false;const video=$('#sim-webcam-preview video');video.srcObject=SIM.webcamStream;if(SIM.webcamCam)attachWebcamOverlay(SIM.webcamCam);$('#sim-webcam-status').textContent='Browser permission granted; sending frames to the tracker.'}if(SIM.videoCam)showVideoSim(SIM.videoCam);if(SIM.scanId)pollAnprSim()}
function simulateHealth(mode){SIM.health=mode;const labels={normal:'Camera feed normal',tamper:'Camera tampering detected (simulation)',loss:'Camera detection lost (simulation)'};SIM.healthEvents.unshift({when:new Date().toLocaleTimeString(),text:labels[mode]});SIM.healthEvents=SIM.healthEvents.slice(0,8);renderHealthSim()}
function renderHealthSim(){const el=$('#sim-health-status'),feed=$('#sim-health-feed');if(!el||!feed)return;const cls=SIM.health==='normal'?'sim-normal':SIM.health==='tamper'?'sim-warning':'sim-danger',label=SIM.health==='normal'?'NORMAL':SIM.health==='tamper'?'SIMULATED TAMPERING':'SIMULATED SIGNAL LOSS',screen=SIM.health==='normal'?'CAMERA FEED NORMAL':SIM.health==='tamper'?'VIDEO OBSTRUCTED':'NO VIDEO SIGNAL',detail=SIM.health==='normal'?'Frames arriving normally':SIM.health==='tamper'?'Lens blocked / image obscured':'Detection input temporarily unavailable';el.innerHTML=`<div class="sim-health-screen ${SIM.health}"><small>SIMULATION · GATE CAMERA</small><strong>${screen}</strong><span>${detail}</span></div><span class="sim-status ${cls}">${label}</span>`;feed.innerHTML=SIM.healthEvents.length?O(SIM.healthEvents,x=>`<div class=sim-event><span>${E(x.text)}</span><small>${E(x.when)}</small></div>`):'<small>No simulation events yet.</small>'}
async function refreshSimVideos(selectId){const v=await api('/api/simulation/videos');SIM.videos=Array.isArray(v)?v:[];for(const id of ['#sim-video-choice','#sim-anpr-choice']){const el=$(id);if(!el)continue;const current=selectId||el.value,empty=SIM.videos.length?'Choose a video clip…':'No clips loaded — add one below';el.innerHTML=`<option value="">${empty}</option>`+O(SIM.videos,x=>`<option value="${E(x.id)}">${E(x.source)} · ${E(x.name)}</option>`);if(SIM.videos.some(x=>x.id===current))el.value=current}}
async function uploadSimVideo(){const f=$('#sim-upload')?.files?.[0],m=$('#sim-library-message');if(!f){m.textContent='Choose a video file first.';return}m.textContent='Adding clip to the local simulation library…';const d=new FormData();d.append('f',f);const r=await api('/api/simulation/videos',{method:'POST',body:d});if(r.error){m.textContent=r.error;return}await refreshSimVideos(r.id);m.textContent=`${r.name} is ready for video tracking or ANPR-only scanning.`}
async function startWebcamSim(){await stopWebcamSim();await stopVideoSim();await stopAnprSim();$('#sim-webcam-status').textContent='Requesting webcam permission…';try{if(!navigator.mediaDevices?.getUserMedia)throw new Error('Webcam access requires localhost or a secure browser connection.');const rate=Math.max(15,Math.min(30,Number($('#sim-webcam-rate')?.value||20))),stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480},frameRate:{ideal:rate,max:30}},audio:false});SIM.webcamStream=stream;$('#sim-webcam-preview').hidden=false;const video=$('#sim-webcam-preview video');video.srcObject=stream;await video.play();const r=await api('/api/simulation/webcam/start',{method:'POST',body:{}});if(r.error)throw new Error(r.error);SIM.webcamCam=r.id;attachWebcamOverlay(r.id);$('#sim-webcam-status').textContent=`Permission granted. Upload target ${rate} FPS; measured AI FPS is shown below.`;sendWebcamFrames()}catch(e){if(SIM.webcamStream)SIM.webcamStream.getTracks().forEach(t=>t.stop());SIM.webcamStream=null;$('#sim-webcam-status').textContent='Webcam could not start: '+(e.message||'permission denied or camera unavailable.')}}
function attachWebcamOverlay(id){const box=$('#sim-webcam-preview'),video=box?.querySelector('video');if(!video)return;video.id='im'+id;if(!$('#cv'+id))box.insertAdjacentHTML('beforeend',`<canvas id=cv${id}></canvas>`)}
function sendWebcamFrames(){const canvas=document.createElement('canvas'),ctx=canvas.getContext('2d');let last=0;const tick=now=>{if(!SIM.webcamCam||!SIM.webcamStream)return;const rate=Math.max(15,Math.min(30,Number($('#sim-webcam-rate')?.value||20)));if(now-last>=1000/rate&&!SIM.webcamBusy){last=now;const video=$('#sim-webcam-preview video');if(video&&video.videoWidth){const w=Math.min(640,video.videoWidth),h=Math.round(video.videoHeight*w/video.videoWidth);canvas.width=w;canvas.height=h;ctx.drawImage(video,0,0,w,h);SIM.webcamBusy=true;const cameraId=SIM.webcamCam;canvas.toBlob(async blob=>{try{if(blob&&SIM.webcamCam===cameraId){const form=new FormData();form.append('frame',blob,'webcam.jpg');const r=await api(`/api/simulation/camera/${cameraId}/frame`,{method:'POST',body:form});if(r.error)$('#sim-webcam-status').textContent=r.error}}catch(e){if($('#sim-webcam-status'))$('#sim-webcam-status').textContent='Webcam frame upload stopped: '+e.message}finally{SIM.webcamBusy=false}},'image/jpeg',.72)}}SIM.webcamRaf=requestAnimationFrame(tick)};SIM.webcamRaf=requestAnimationFrame(tick)}
async function stopWebcamSim(){if(SIM.webcamRaf)cancelAnimationFrame(SIM.webcamRaf);SIM.webcamRaf=0;if(SIM.webcamStream)SIM.webcamStream.getTracks().forEach(t=>t.stop());SIM.webcamStream=null;const id=SIM.webcamCam;SIM.webcamCam=null;if(id){delete TRACKS[id];try{await api(`/api/simulation/camera/${id}/stop`,{method:'POST'})}catch(e){}}const p=$('#sim-webcam-preview');if(p){p.hidden=true;p.innerHTML='<video id="sim-webcam-video" autoplay muted playsinline></video>'}const s=$('#sim-webcam-status');if(s)s.textContent='Webcam stopped.'}
async function startVideoSim(){await stopWebcamSim();await stopAnprSim();const key=$('#sim-video-choice')?.value;if(!key)return $('#sim-video-info').textContent='Choose or upload a video clip first.';const r=await api('/api/simulation/video/start',{method:'POST',body:{video:key}});if(r.error){$('#sim-video-info').textContent=r.error;return}SIM.videoCam=r.id;showVideoSim(r.id);$('#sim-video-info').textContent='Starting detector… measured AI FPS will appear here.'}
function showVideoSim(id){const box=$('#sim-video-preview');if(!box)return;box.hidden=false;box.className='v';box.innerHTML=`<img id=im${id} alt="Simulation video with tracked boxes" src="/stream/${id}?t=${T}"><canvas id=cv${id}></canvas><div class=vwait id=sim-wait-${id}>Waiting for the first analyzed frame…</div>`;const im=$('#im'+id);im.onload=()=>{const w=$(`#sim-wait-${id}`);if(w)w.hidden=true}}
async function stopVideoSim(){const id=SIM.videoCam;SIM.videoCam=null;if(id){delete TRACKS[id];try{await api(`/api/simulation/camera/${id}/stop`,{method:'POST'})}catch(e){}}const box=$('#sim-video-preview'),info=$('#sim-video-info');if(box){box.hidden=true;box.innerHTML=''}if(info)info.textContent='Video simulation stopped.'}
function videoUrl(key){const [kind,name]=String(key||'').split(':');return `/api/simulation/files/${encodeURIComponent(kind)}/${encodeURIComponent(name)}?t=${encodeURIComponent(T)}`}
async function startAnprSim(){await stopWebcamSim();await stopVideoSim();if(SIM.scanId)await stopAnprSim();const key=$('#sim-anpr-choice')?.value;if(!key)return $('#sim-anpr-status').textContent='Choose or upload a video clip first.';const r=await api('/api/simulation/anpr/start',{method:'POST',body:{video:key}});if(r.error){$('#sim-anpr-status').textContent=r.error;return}SIM.scanId=r.id;$('#sim-anpr-player').hidden=false;$('#sim-anpr-player').innerHTML=`<video controls autoplay muted playsinline src="${videoUrl(key)}"></video>`;$('#sim-anpr-confirmed').innerHTML='<tr><td colspan=4 class=sim-help>Waiting for plate detections…</td></tr>';$('#sim-anpr-raw').innerHTML='<tr><td colspan=4 class=sim-help>No OCR candidates yet.</td></tr>';SIM.scanTimer=setInterval(pollAnprSim,900);pollAnprSim()}
async function pollAnprSim(){if(!SIM.scanId)return;const s=await api(`/api/simulation/anpr/${SIM.scanId}`);if(s.error){$('#sim-anpr-status').textContent=s.error;return}$('#sim-anpr-status').textContent=`${String(s.status).toUpperCase()} · ${s.processed_frames||0}${s.total_frames?` / ${s.total_frames}`:''} frames · ${s.scan_fps||0} scan FPS${s.error?` · ${s.error}`:''}`;const progress=$('#sim-anpr-progress');if(progress)progress.style.width=(s.progress||0)+'%';if(s.confirmed?.length)$('#sim-anpr-confirmed').innerHTML=O(s.confirmed,x=>`<tr><td>#${x.track}</td><td><b>${E(x.plate)}</b></td><td>${x.votes}</td><td>${x.last_frame}</td></tr>`);if(s.raw_reads?.length)$('#sim-anpr-raw').innerHTML=O(s.raw_reads.slice().reverse().map(x=>`<tr><td>${x.frame}</td><td>#${x.track}</td><td>${E(x.raw)}</td><td>${x.confidence}</td></tr>`))||'<tr><td colspan=4>No OCR candidates found.</td></tr>';if(['complete','cancelled','error'].includes(s.status)){clearInterval(SIM.scanTimer);SIM.scanTimer=0;if(s.status==='complete'&&!s.raw_reads?.length)$('#sim-anpr-raw').innerHTML='<tr><td colspan=4>No plate candidates detected in this clip.</td></tr>'}}
async function stopAnprSim(){const id=SIM.scanId;SIM.scanId=null;if(SIM.scanTimer)clearInterval(SIM.scanTimer);SIM.scanTimer=0;if(id){try{await api(`/api/simulation/anpr/${id}/stop`,{method:'POST'})}catch(e){}}}
async function closeSimulation(){await stopWebcamSim();await stopVideoSim();await stopAnprSim()}
window.addEventListener('beforeunload',()=>{for(const id of [SIM.webcamCam,SIM.videoCam])if(id)navigator.sendBeacon(`/api/simulation/camera/${id}/stop?t=${encodeURIComponent(T)}`,new Blob([],{type:'application/octet-stream'}));if(SIM.webcamStream)SIM.webcamStream.getTracks().forEach(t=>t.stop());if(SIM.scanId)navigator.sendBeacon(`/api/simulation/anpr/${SIM.scanId}/stop?t=${encodeURIComponent(T)}`,new Blob([],{type:'application/octet-stream'}))});
async function alerts(){const a=await api('/api/alerts');$('#m').innerHTML='<h2>Alerts</h2><div class=k><table><tr><th>Time</th><th>Source</th><th>Type</th><th>Details</th><th>Evidence</th></tr>'+O(a,al)+'</table></div>'}
async function anpr(){const [r,p]=await Promise.all([api('/api/reads'),api('/api/plates')]);$('#m').innerHTML=`<h2>ANPR Readings</h2><div class=g>
<div class=k><h3>All Readings Before Voting</h3><small>Every OCR result, including wrong ones</small><table><tr><th>Time</th><th>Source</th><th>Track</th><th>Reading</th><th>Conf.</th><th>Crop</th></tr>${O(r,x=>`<tr><td>${tm(x.ts)}</td><td>${E(x.cname)}</td><td>#${x.track}</td><td><b>${E(x.raw)}</b></td><td>${x.conf}</td><td>${x.crop?`<img class=crop src="${IM(x.crop)}">`:''}</td></tr>`)}</table></div>
<div class=k><h3>Confirmed Plates After Voting</h3><small>Reads are combined per vehicle track</small><table><tr><th>Time</th><th>Source</th><th>Track</th><th>Plate</th><th>Reads</th></tr>${O(p,x=>`<tr><td>${tm(x.ts)}</td><td>${E(x.cname)}</td><td>#${x.track}</td><td><b>${E(x.plate)}</b></td><td>${x.votes}</td></tr>`)}</table></div></div>`}
async function incidents(){const a=await api('/api/incidents');$('#m').innerHTML='<h2>Incidents</h2><div class=k><table><tr><th>ID</th><th>Title</th><th>Status</th><th>Opened</th><th></th></tr>'+O(a,x=>`<tr><td>#${x.id}</td><td class=cap>${E(x.title.replace(/_/g,' '))}</td><td class="cap ${x.status=='open'?'bad':'ok'}">${x.status}</td><td>${new Date(x.ts*1e3).toLocaleString()}</td><td><button onclick="tl(${x.id})">Timeline</button> <a href="/api/incidents/${x.id}/export?t=${T}"><button>Export Evidence</button></a> ${x.status=='open'?`<button onclick="cl(${x.id})">Close</button>`:''}</td></tr>`)+'</table></div><div id=tl></div>'}
async function tl(i){const a=await api('/api/incidents/'+i);$('#tl').innerHTML='<h3 style="margin-top:16px">Timeline</h3><div class=k><table>'+O(a,al)+'</table></div>'}
async function cl(i){await api(`/api/incidents/${i}/close`,{method:'POST'});incidents()}
async function evidence(){const a=await api('/api/alerts?snap=1');$('#m').innerHTML='<h2>Evidence</h2>'+(a.length?'':'<div class=k>No Evidence Yet. Snapshots Are Saved Automatically With Every Alert. You Can Also Press Save Snapshot In Live View.</div>')+'<div class=th>'+
 O(a,x=>`<div class=k style=padding:8px><a target=_blank href="${IM(x.snap)}"><img src="${IM(x.snap)}"></a><div class=cap style=margin-top:6px><b class=${x.sev}>${x.type.replace(/_/g,' ')}</b></div><small>${E(x.cname||x.cam)} | ${new Date(x.ts*1e3).toLocaleString()}</small><br><small>${E(x.info)}</small></div>`)+'</div>'}
async function plan(){const c=await api('/api/cams');$('#m').innerHTML=`<h2>Floor Plan</h2><div class=k><input type=file id=fpf> <button onclick="upf()">Upload Floor Plan</button> <select id=sc>${O(c,x=>`<option value=${x.id}>${E(x.name)}</option>`)}</select> <small>Pick A Source, Then Click The Plan To Place It</small><br><div style="position:relative;display:inline-block;margin-top:8px"><img src="/floorplan?t=${T}" onclick="place(event)">${O(c.filter(x=>x.fx!=null),x=>`<span style="position:absolute;left:${x.fx*100}%;top:${x.fy*100}%;transform:translate(-50%,-50%);background:${x.status=='online'?'#3fb950':'#f85149'};color:#000;padding:2px 8px;border-radius:10px">${E(x.name)}</span>`)}</div></div>`}
async function upf(){const f=new FormData();f.append('f',$('#fpf').files[0]);await api('/api/floorplan',{method:'POST',body:f});plan()}
async function place(e){const m=e.target;await api(`/api/cams/${$('#sc').value}/pos`,{method:'POST',body:{fx:e.offsetX/m.clientWidth,fy:e.offsetY/m.clientHeight}});plan()}
const EV=['intrusion','fence_exit','loitering','plate','reid','tamper','signal_loss','signal_restored'],SV=['','info','low','medium','high','critical'],cap=s=>s.replace(/_/g,' ').replace(/^./,c=>c.toUpperCase());
async function rules(){const [c,r,s]=await Promise.all([api('/api/cams'),api('/api/rules'),api('/api/settings')]),cn=Object.fromEntries(c.map(x=>[x.id,x.name])),ck=v=>v?'checked':'';
$('#m').innerHTML=`<h2>Rules & Settings</h2><div class=g><section class=k><h3>Detection settings</h3><div class=form-grid>
<label class=field>Loitering threshold (seconds)<input id=sd type=number min=1 max=3600 value=${s.dwell_s}></label><label class=field>AI processing target (FPS)<input id=sa type=number min=15 max=120 step=5 value=${s.ai_fps}></label>
<label class=field>AI image size<select id=si>${O([320,480,640,960],v=>`<option ${v==s.imgsz?'selected':''}>${v}</option>`)}</select></label><label class=field>Stream maximum width<input id=sw type=number min=320 max=1920 value=${s.stream_w}></label>
<label class=field>MQTT host<input id=mh value="${E(s.mqtt_host)}" placeholder="Optional event destination"></label><label class=field>MQTT port<input id=mp type=number min=1 max=65535 value=${s.mqtt_port}></label></div>
<div class=check-row><label><input type=checkbox id=fa ${ck(s.anpr)}> Enable ANPR (every six YOLO frames)</label><label><input type=checkbox id=fr ${ck(s.reid)}> Enable person Re-ID</label></div><button class=pri onclick="sets()">Save settings</button>
<p><small>Video display and AI processing run in the background independently. The selected rate is a target; Live View shows measured AI FPS. Smaller image size and disabled auxiliary processing reduce workload on slower computers.</small></p></section>
<section class=k><h3>Rule engine</h3><div class=rule-list>${r.length?O(r,x=>`<div class=rule-item><div class=rule-summary>When <b>${cap(x.etype)}</b> on <b>${x.cam?E(cn[x.cam]||'Unknown source'):'Any source'}</b> occurs ${x.cnt} time(s) within ${x.win}s. Severity: ${cap(x.sev||'unchanged')}${x.incident?'; open an incident':''}.</div><button onclick="del('rules',${x.id})">Remove</button></div>`):'<small>No rules yet. Add one below to customize event severity or incident creation.</small>'}</div>
<div class=rule-builder><h4>Create a rule</h4><div class=rule-form-grid><label class=field>Event<select id=re>${O(EV,v=>`<option value=${v}>${cap(v)}</option>`)}</select></label><label class=field>Source<select id=rc><option value="">Any source</option>${O(c,x=>`<option value=${x.id}>${E(x.name)}</option>`)}</select></label>
<label class=field>Occurrences<input id=rn type=number min=1 value=1></label><label class=field>Within (seconds)<input id=rw type=number min=1 value=60></label><label class=field>Set severity<select id=rs>${O(SV,v=>`<option value="${v}">${cap(v||'unchanged')}</option>`)}</select></label><label class="field check-field"><span>Incident action</span><span><input type=checkbox id=ri> Open incident</span></label></div><button class=pri onclick="addr()">Add rule</button></div></section></div>`}
async function admin(){const [u,v]=await Promise.all([api('/api/users'),api('/api/verify')]);
$('#m').innerHTML=`<h2>Users & Integrity</h2><div class=g>
<section class=k><h3>Users</h3><div class=rule-list>${O(u,x=>`<div class=rule-item><span>${E(x.u)} <small class=cap>${x.role}</small></span></div>`)||'<small>No users</small>'}</div><hr><div class=rule-form-grid><label class=field>Username<input id=nu placeholder="Username"></label><label class=field>Password<input id=np type=password placeholder="At least 8 characters"></label><label class=field>Role<select id=nr>${O(['viewer','operator','admin'],x=>`<option value=${x}>${cap(x)}</option>`)}</select></label></div><button class=pri onclick="addu()">Add user</button></section>
<section class=k><h3>Tamper-evident SHA-256 hash chain / local ledger</h3><div id=verify-status aria-live=polite>${integrityResult(v)}</div><button class=pri id=verify-chain-btn onclick="verifyIntegrity()">Verify Hash Chain</button><p><small>Checks alert hashes, ledger links, and both chains. This is a local tamper-evident ledger, not a blockchain.</small></p></section></div>`}
function integrityResult(v){return v.ok?`<span class=ok>Verification passed: ${v.alerts_checked} alerts and ${v.ledger_checked} ledger entries checked.</span>`:`<span class=bad>Verification failed: ${E((v.bad||[]).join(', '))}</span>`}
async function verifyIntegrity(){const b=$('#verify-chain-btn'),s=$('#verify-status');if(b)b.disabled=true;if(s)s.textContent='Verifying alert hashes and local ledger…';try{const v=await api('/api/verify');if(s)s.innerHTML=integrityResult(v)}catch(e){if(s)s.textContent='Verification could not complete: '+e.message}finally{if(b)b.disabled=false}}
function account(){$('#m').innerHTML=`<h2>Account</h2><section class=k style="max-width:460px"><h3>Change password</h3><div class=rule-form-grid><label class="field wide">Current password<input id=po type=password autocomplete=current-password placeholder="Current password"></label><label class="field wide">New password<input id=pn type=password autocomplete=new-password placeholder="At least 8 characters"></label></div><button class=pri onclick="pw()">Change password</button><div id=pwmsg class=metric-note role=status></div></section>`}
async function del(k,i){await api(`/api/${k}/${i}`,{method:'DELETE'});draw()}
async function addr(){await api('/api/rules',{method:'POST',body:{etype:$('#re').value,cam:$('#rc').value,sev:$('#rs').value,incident:$('#ri').checked?1:0,cnt:$('#rn').value,win:$('#rw').value}});rules()}
async function addu(){const r=await api('/api/users',{method:'POST',body:{u:$('#nu').value,p:$('#np').value,role:$('#nr').value}});if(r.error)alert(r.error);admin()}
async function sets(){await api('/api/settings',{method:'POST',body:{ai_fps:$('#sa').value,dwell_s:$('#sd').value,imgsz:$('#si').value,stream_w:$('#sw').value,anpr:$('#fa').checked?1:0,reid:$('#fr').checked?1:0,mqtt_host:$('#mh').value,mqtt_port:$('#mp').value}});rules()}
async function pw(){const r=await api('/api/password',{method:'POST',body:{old:$('#po').value,new:$('#pn').value}});alert(r.ok?'Password Changed':r.error)}
init()
</script>
"""

def ensure_models():
    if not os.path.exists(YOLO_PATH):
        print('Downloading YOLO11n weights (one-time)...'); cwd = os.getcwd(); os.chdir(MD)
        try: YOLO('yolo11n.pt')
        finally: os.chdir(cwd)

def prepare_models():
    global MODEL_ERROR
    try: ensure_models()
    except Exception as e: MODEL_ERROR = repr(e); print('YOLO model setup failed:', MODEL_ERROR)
    finally: MODEL_READY.set()

RUNTIME_STARTED=False; RUNTIME_LOCK=threading.Lock()
def start_background_services():
    global RUNTIME_STARTED
    with RUNTIME_LOCK:
        if RUNTIME_STARTED: return
        RUNTIME_STARTED=True
    mq_connect(); threading.Thread(target=prepare_models, daemon=True).start()
    for _ in range(2 if (os.cpu_count() or 2) < 8 else 3): threading.Thread(target=worker, daemon=True).start()
    for r in q('select * from cams'): start(r)

if __name__ == '__main__':
    start_background_services()
    print('device:', DEVICE_LABEL, '| ANPR:', 'on' if OCR else 'off')
    port = int(os.getenv('PORT', 8000)); print(f'\nOpen  http://127.0.0.1:{port}  in your browser (opening it for you now)\n')
    threading.Timer(2, lambda: webbrowser.open(f'http://127.0.0.1:{port}')).start()
    app.run(os.getenv('HOST', '127.0.0.1'), port, threaded=True)
