/* fraud-transaction-detector — LogReg on engineered numeric features, trained in-browser on synthetic data. */
(function(){
'use strict';
const $=id=>document.getElementById(id);
let model=null,trained=false,metrics=null;
const DIM=12;
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function seededShuffle(arr,seed){const r=mulberry32(seed);for(let i=arr.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[arr[i],arr[j]]=[arr[j],arr[i]]}return arr}
function createModel(){
  const m=tf.sequential();
  m.add(tf.layers.dense({units:1,inputShape:[DIM],activation:'sigmoid',kernelRegularizer:tf.regularizers.l2({l2:0.005})}));
  m.compile({optimizer:'adam',loss:'binaryCrossentropy',metrics:['accuracy']});
  return m;
}
async function fitRows(rows){
  const m=createModel();
  const x=tf.tensor2d(rows.map(r=>featurize(r)));
  const y=tf.tensor1d(rows.map(r=>r.fraud));
  await m.fit(x,y,{epochs:120,batchSize:64,shuffle:true,verbose:0});
  x.dispose();y.dispose();
  return m;
}
async function predictRows(m,rows){
  const x=tf.tensor2d(rows.map(r=>featurize(r)));
  const pv=await m.predict(x).data();x.dispose();
  return Array.from(pv);
}
async function crossValidate(prog){
  const data=TRAIN_TX.slice();seededShuffle(data,7);
  const FOLDS=5,oof=new Array(data.length);
  for(let k=0;k<FOLDS;k++){
    const te=data.filter((_,i)=>i%FOLDS===k),tr=data.filter((_,i)=>i%FOLDS!==k);
    const m=await fitRows(tr);
    const pv=await predictRows(m,te);
    te.forEach((d,i)=>oof[data.indexOf(d)]=pv[i]);
    m.dispose();
    if(prog)prog(`Cross-validating… fold ${k+1}/${FOLDS}`);
  }
  let tp=0,tn=0,fp=0,fn=0;
  oof.forEach((p,i)=>{const c=p>=0.5?1:0,y=data[i].fraud;if(c===1&&y===1)tp++;else if(c===0&&y===0)tn++;else if(c===1)fp++;else fn++});
  const prec=tp/(tp+fp||1),rec=tp/(tp+fn||1);
  return{acc:(tp+tn)/data.length,prec,rec,f1:2*prec*rec/(prec+rec||1),n:data.length,tp,tn,fp,fn};
}
async function train(prog){
  metrics=await crossValidate(prog);
  model=await fitRows(TRAIN_TX);
  trained=true;renderMetrics();
}
function renderMetrics(){
  if(!metrics)return;
  $('m-acc').textContent=(metrics.acc*100).toFixed(1)+'%';
  $('m-prec').textContent=(metrics.prec*100).toFixed(1)+'%';
  $('m-rec').textContent=(metrics.rec*100).toFixed(1)+'%';
  $('m-conf').innerHTML=`<tr><th></th><th>Predicted fraud</th><th>Predicted legit</th></tr>
  <tr><th>Actually fraud</th><td class="cm-ok">${metrics.tp}</td><td class="cm-bad">${metrics.fn}</td></tr>
  <tr><th>Actually legit</th><td class="cm-bad">${metrics.fp}</td><td class="cm-ok">${metrics.tn}</td></tr>`;
}
const FACTOR_LABELS=['Large amount','Unusual hour','Brand-new payee','Far from home','International','Burst of transactions','Online channel','Shopping category','Electronics category','Transfer category','ATM category','Travel category'];
async function scoreTx(r){
  const v=featurize(r);
  const t=tf.tensor2d([v]);
  const p=(await model.predict(t).data())[0];t.dispose();
  // factor contributions: weight * feature value
  const w=model.layers[0].getWeights()[0].dataSync();
  const contribs=v.map((val,i)=>({f:FACTOR_LABELS[i],c:w[i]*val,val})).filter(x=>Math.abs(x.c)>0.02).sort((a,b)=>b.c-a.c);
  return {p,contribs};
}
function readForm(){
  return {
    amount:parseFloat($('f-amount').value)||0,
    hour:parseInt($('f-hour').value)||12,
    category:$('f-category').value,
    payeeAgeDays:parseInt($('f-payee').value)||365,
    distanceKm:parseFloat($('f-distance').value)||0,
    channel:$('f-channel').value,
    intl:$('f-intl').checked?1:0,
    txLastHour:parseInt($('f-velocity').value)||1,
  };
}
async function analyze(){
  if(!trained){pkToast('Model is still loading, one moment…');return}
  const r=readForm();
  const {p,contribs}=await scoreTx(r);
  $('result').classList.remove('hidden');
  const label=p>=0.7?['High fraud risk','bad']:p>=0.4?['Review recommended','warn']:['Looks legitimate','ok'];
  $('verdict').innerHTML=`<span class="pill ${label[1]}">${label[0]}</span>`;
  $('prob-num').textContent=(p*100).toFixed(1)+'%';
  $('prob-meter').style.width=(p*100)+'%';
  $('prob-meter').style.background=p>=0.4?'var(--bad)':'var(--ok)';
  $('prob-note').textContent='Risk score from 12 transaction signals. Thresholds: ≥70% high risk, 40-70% review, <40% likely fine.';
  $('factors').innerHTML=contribs.slice(0,6).map(c=>`<div class="factor"><span>${c.f}</span><span class="mono ${c.c>0?'bad-t':'ok-t'}">${c.c>0?'+':''}${c.c.toFixed(2)}</span></div>`).join('')||'<p class="hint">No strong factors either way.</p>';
}
async function batch(){
  if(!trained){pkToast('Model is still loading, one moment…');return}
  const raw=$('batch-in').value.trim();if(!raw){pkToast('Paste CSV rows first');return}
  const rows=[];
  for(const line of raw.split('\n')){
    const p=line.split(',').map(x=>x.trim());
    if(p.length<8||isNaN(parseFloat(p[0])))continue;
    rows.push({amount:parseFloat(p[0]),hour:parseInt(p[1]),category:p[2].toLowerCase(),payeeAgeDays:parseInt(p[3]),distanceKm:parseFloat(p[4]),channel:p[5].toLowerCase(),intl:/1|yes|true/i.test(p[6])?1:0,txLastHour:parseInt(p[7])||1});
  }
  if(!rows.length){pkToast('No valid rows — need amount,hour,category,payeeAgeDays,distanceKm,channel,intl,txLastHour');return}
  const out=[];
  for(const r of rows){const {p}=await scoreTx(r);out.push({r,p})}
  $('batch-out').classList.remove('hidden');
  $('batch-tbl').innerHTML=out.map((o,i)=>{
    const lab=o.p>=0.7?'<span class="pill bad">High risk</span>':o.p>=0.4?'<span class="pill warn">Review</span>':'<span class="pill ok">Legit</span>';
    return `<tr><td>${i+1}</td><td class="mono">₹${o.r.amount.toLocaleString()}</td><td>${o.r.category}</td><td>${lab}</td><td class="mono">${(o.p*100).toFixed(0)}%</td></tr>`}).join('');
  const nH=out.filter(o=>o.p>=0.7).length,nR=out.filter(o=>o.p>=0.4&&o.p<0.7).length;
  $('batch-sum').innerHTML=`<span class="pill bad">${nH} high risk</span> <span class="pill warn">${nR} review</span> <span class="pill ok">${out.length-nH-nR} legit</span>`;
  $('batch-dl').classList.remove('hidden');
  $('batch-dl').onclick=()=>pkDownload('fraud-screening.csv',pkCsv([['amount','hour','category','risk','score'],...out.map(o=>[o.r.amount,o.r.hour,o.r.category,o.p>=0.7?'high':o.p>=0.4?'review':'legit',o.p.toFixed(3)])]),'text/csv');
}
function base64ToBuf(b64){const bin=atob(b64);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u.buffer}
function bufToBase64(buf){const u=new Uint8Array(buf);let s='';for(let i=0;i<u.length;i+=8192)s+=String.fromCharCode.apply(null,u.subarray(i,i+8192));return btoa(s)}
async function exportRun(){
  metrics=await crossValidate(t=>{$('train-prog').textContent=t});
  model=await fitRows(TRAIN_TX);
  const art=await model.save(tf.io.withSaveHandler(async a=>({modelTopology:a.modelTopology,weightSpecs:a.weightSpecs,weightData:bufToBase64(a.weightData)})));
  art.metrics=metrics;art.dim=DIM;
  window.EXPORT=JSON.stringify(art);
  $('train-prog').textContent='EXPORT READY';
}
function readyUI(source){
  trained=true;
  $('train-prog').innerHTML='<span class="pill ok">Model ready</span> '+source;
  renderMetrics();
}
async function boot(){
  if(location.search.includes('export'))return exportRun();
  try{
    const r=await fetch('weights.json');
    if(r.ok){
      const art=await r.json();
      art.weightData=base64ToBuf(art.weightData);
      model=await tf.loadLayersModel(tf.io.fromMemory(art));
      metrics=art.metrics||null;
      readyUI('Pre-trained weights loaded — runs fully offline.');
      return;
    }
  }catch(e){console.warn('weights load failed, training live',e)}
  train(t=>{$('train-prog').textContent=t}).then(()=>readyUI('Trained live in your browser — cross-validated metrics below.')).catch(e=>{$('train-prog').textContent='Model failed: '+e.message;console.error(e)});
}
document.addEventListener('DOMContentLoaded',()=>{
  $('analyze').onclick=analyze;
  $('batch-btn').onclick=batch;
  $('sample').onclick=()=>{
    const samples=[
      {amount:4200,hour:2,category:'electronics',payeeAgeDays:1,distanceKm:3200,channel:'online',intl:true,txLastHour:5},
      {amount:340,hour:13,category:'groceries',payeeAgeDays:900,distanceKm:3,channel:'instore',intl:false,txLastHour:1},
      {amount:8900,hour:23,category:'transfer',payeeAgeDays:0,distanceKm:800,channel:'online',intl:true,txLastHour:4},
    ];
    const s=samples[Math.floor(Math.random()*samples.length)];
    $('f-amount').value=s.amount;$('f-hour').value=s.hour;$('f-category').value=s.category;
    $('f-payee').value=s.payeeAgeDays;$('f-distance').value=s.distanceKm;$('f-channel').value=s.channel;
    $('f-intl').checked=s.intl;$('f-velocity').value=s.txLastHour;
    analyze();
  };
  $('retrain').onclick=async()=>{
    const rb=$('retrain');if(rb.disabled)return;rb.disabled=true;
    $('train-prog').textContent='Retraining in your browser…';
    try{await train(t=>{$('train-prog').textContent=t});readyUI('Retrained live — cross-validated metrics below.')}catch(e){$('train-prog').textContent='Training failed: '+e.message}
    rb.disabled=false;
  };
  boot();
});
})();
