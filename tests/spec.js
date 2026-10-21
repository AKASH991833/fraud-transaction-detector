module.exports=async(page)=>{
  await page.waitForFunction(()=>document.querySelector('#train-prog')&&/ready|trained/.test(document.querySelector('#train-prog').textContent),{timeout:120000});
  // obvious fraud sample
  await page.evaluate(()=>{
    document.getElementById('f-amount').value='4200';document.getElementById('f-hour').value='2';
    document.getElementById('f-category').value='electronics';document.getElementById('f-payee').value='1';
    document.getElementById('f-distance').value='3200';document.getElementById('f-channel').value='online';
    document.getElementById('f-intl').checked=true;document.getElementById('f-velocity').value='5';
  });
  await page.click('#analyze');
  await page.waitForSelector('#result:not(.hidden)',{timeout:20000});
  const v1=await page.$eval('#verdict',e=>e.textContent);
  if(!/risk|Review/i.test(v1))throw new Error('fraud not flagged: '+v1);
  // clean grocery sample
  await page.evaluate(()=>{
    document.getElementById('f-amount').value='180';document.getElementById('f-hour').value='13';
    document.getElementById('f-category').value='groceries';document.getElementById('f-payee').value='900';
    document.getElementById('f-distance').value='2';document.getElementById('f-channel').value='instore';
    document.getElementById('f-intl').checked=false;document.getElementById('f-velocity').value='1';
    document.getElementById('prob-num').textContent='—';
  });
  await page.click('#analyze');
  await page.waitForFunction(()=>document.querySelector('#prob-num').textContent!=='—',{timeout:20000});
  const v2=await page.$eval('#verdict',e=>e.textContent);
  if(!/legitimate/i.test(v2))throw new Error('legit misflagged: '+v2);
  const acc=await page.$eval('#m-acc',e=>e.textContent);
  const fac=await page.$$eval('#factors .factor',e=>e.length);
  return `fraud+legit scored correctly, ${fac} factors shown, CV acc=${acc}`;
};
