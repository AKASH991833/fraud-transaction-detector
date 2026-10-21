/* Synthetic transaction data generator — seeded, deterministic.
   Fraud patterns injected with clear signatures so a small model can learn them.
   HONEST: this data is synthetic; the model learns these injected patterns, not real fraud. */
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
const CATS=['groceries','fuel','dining','shopping','travel','electronics','transfer','atm'];
function genTransactions(n,seed){
  const rnd=mulberry32(seed);const rows=[];
  const pick=a=>a[Math.floor(rnd()*a.length)];
  for(let i=0;i<n;i++){
    const fraud=rnd()<0.12;
    let r;
    if(fraud){
      // fraud signatures: big amounts, odd hours, far away, brand-new payees, online international, bursts
      const pattern=Math.floor(rnd()*4);
      r={
        amount:Math.round((pattern===0?1500+rnd()*8500:200+rnd()*3000)),
        hour:pattern===1?Math.floor(rnd()*5):Math.floor(rnd()*24),
        category:pattern===2?'electronics':(pattern===3?'transfer':pick(CATS)),
        payeeAgeDays:pattern===2?Math.floor(rnd()*3):Math.floor(rnd()*400),
        distanceKm:pattern===1?200+rnd()*4800:Math.round(rnd()*3000),
        channel:pattern===3||rnd()<0.7?'online':'instore',
        intl:pattern===3?1:(rnd()<0.55?1:0),
        txLastHour:pattern===0?3+Math.floor(rnd()*6):1+Math.floor(rnd()*3),
        fraud:1
      };
    }else{
      r={
        amount:Math.round(5+rnd()*rnd()*400),
        hour:7+Math.floor(rnd()*14),
        category:pick(CATS),
        payeeAgeDays:30+Math.floor(rnd()*2000),
        distanceKm:Math.round(rnd()*rnd()*40),
        channel:rnd()<0.5?'online':'instore',
        intl:rnd()<0.03?1:0,
        txLastHour:1,
        fraud:0
      };
    }
    rows.push(r);
  }
  return rows;
}
const TRAIN_TX=genTransactions(1200,42);
const FEATURES=['amount','hour','payeeAgeDays','distanceKm','intl','txLastHour','online','cat_shopping','cat_electronics','cat_transfer','cat_atm','cat_travel'];
function featurize(r){
  return [
    Math.min(1,r.amount/10000),
    Math.abs(r.hour-14)/14,          // distance from mid-day
    1-Math.min(1,r.payeeAgeDays/2000),
    Math.min(1,r.distanceKm/5000),
    r.intl,
    Math.min(1,(r.txLastHour-1)/8),
    r.channel==='online'?1:0,
    r.category==='shopping'?1:0,
    r.category==='electronics'?1:0,
    r.category==='transfer'?1:0,
    r.category==='atm'?1:0,
    r.category==='travel'?1:0
  ];
}
