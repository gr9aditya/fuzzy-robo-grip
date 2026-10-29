'use strict';
const $ = id => document.getElementById(id);
const escapeHTML = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (n, digits=2) => Number(n).toLocaleString('en-US', {minimumFractionDigits:digits, maximumFractionDigits:digits});
const force = n => fmt(n, n < 0.1 ? 4 : n < 1 ? 3 : 2);
let currentResult = null;
let provenance = '';
let profileList = [];
const numericKeys = ['mass_g','mu','mass_uncertainty_pct','friction_uncertainty_pct','limit_n'];
const fieldNames = {mass_g:'Mass',mu:'Contact friction coefficient',mass_uncertainty_pct:'Mass uncertainty',friction_uncertainty_pct:'Friction uncertainty',limit_n:'Allowable total grip force',name:'Object name',contact:'Contact description'};

function step(number){
  for(let i=1;i<=3;i++){
    $('step'+i).hidden=i!==number;
    $('s'+i).className=i===number?'current':i<number?'done':'';
    if(i===number) $('s'+i).setAttribute('aria-current','step'); else $('s'+i).removeAttribute('aria-current');
  }
  window.scrollTo({top:0,behavior:'instant'});
  const heading=$('step'+number).querySelector('h1');
  heading.tabIndex=-1; heading.focus({preventScroll:true});
}

function icon(id){
  const drawings = {
    glass:'<path d="M15 12h34l-4 47H19z"/><path d="M16 18h32M21 53h22"/>',
    filled_glass:'<path d="M15 12h34l-4 47H19z"/><path d="M18 31h28l-2 25H20z" fill="#dbe9e0"/><path d="M18 31q7-4 14 0t14 0"/>',
    phone:'<rect x="17" y="6" width="31" height="57" rx="5"/><path d="M26 12h13M29 57h7"/>',
    box:'<path d="M8 25l24-13 24 13v27L32 65 8 52zM8 25l24 13 24-13M32 38v27M20 19l24 13"/>'
  };
  return `<svg viewBox="0 0 64 72" aria-hidden="true">${drawings[id]||drawings.box}</svg>`;
}

function choose(profile){
  $('objectForm').reset();
  for(const field of $('objectForm').elements) field.removeAttribute('aria-invalid');
  $('formError').textContent='';
  $('name').value=profile.name;
  $('mass_g').value=profile.mass_g ?? '';
  $('mu').value=profile.mu ?? '';
  $('contact').value=profile.contact ?? '';
  $('reviewName').textContent=profile.name;
  provenance=profile.note;
  $('profileNote').textContent=profile.note;
  currentResult=null;
  step(2);
}

async function loadProfiles(){
  try{
    const response=await fetch('/api/profiles');
    if(!response.ok) throw new Error('Could not load example profiles.');
    profileList=await response.json();
    for(const profile of profileList){
      const button=document.createElement('button');
      button.type='button';button.className='profile';
      button.innerHTML=`<span class="object-icon">${icon(profile.id)}</span><strong>${escapeHTML(profile.name)}</strong><span class="meta">Example profile · ${fmt(profile.mass_g,0)} g</span><span class="use">Review this object →</span>`;
      button.addEventListener('click',()=>choose(profile));
      $('profiles').append(button);
    }
  }catch(error){$('chooseError').textContent=error.message+' Check that the Python app is running, then refresh.';}
}

$('customButton').onclick=()=>{
  const name=$('customName').value.trim();
  if(!name){$('chooseError').textContent='Enter an object name to continue.';$('customName').focus();return;}
  $('chooseError').textContent='';
  choose({name,note:'Custom object. Enter mass and contact assumptions; the name alone does not determine them.'});
};
$('customName').onkeydown=e=>{if(e.key==='Enter'){$('customButton').click();}};
$('back1').onclick=()=>step(1);
for(const id of ['back2','adjust']) $(id).onclick=()=>step(2);
$('another').onclick=()=>step(1);
$('name').oninput=()=>{$('reviewName').textContent=$('name').value;};
$('objectForm').addEventListener('input',e=>{currentResult=null;e.target.removeAttribute('aria-invalid');$('formError').textContent='';});

function chart(degrees, bound){
  const colors=['#1d6553','#6976bd','#b26d37'];
  const paths=['M30 25 L180 125 L330 125','M30 125 L180 25 L330 125','M30 125 L180 125 L330 25'];
  const x=30+bound/50*300;
  return `<svg viewBox="0 0 360 165" role="img" aria-label="Membership degrees at ${bound} percent: low ${fmt(degrees[0])}, medium ${fmt(degrees[1])}, high ${fmt(degrees[2])}"><path d="M30 20v105h300" fill="none" stroke="#c6d1c8"/><text x="8" y="29" font-size="10" fill="#62756e">1</text><text x="8" y="128" font-size="10" fill="#62756e">0</text>${paths.map((p,i)=>`<path d="${p}" fill="none" stroke="${colors[i]}" stroke-width="2.5"/>`).join('')}<path d="M${x} 20V125" stroke="#18352f" stroke-dasharray="4 3"/>${degrees.map((d,i)=>`<circle cx="${x}" cy="${125-d*100}" r="4" fill="${colors[i]}" stroke="white"/>`).join('')}<text x="24" y="148" font-size="11" fill="#62756e">0%</text><text x="169" y="148" font-size="11" fill="#62756e">25%</text><text x="316" y="148" font-size="11" fill="#62756e">50%</text></svg><div class="membership-legend">${degrees.map((d,i)=>`<span style="color:${colors[i]}">${['Low','Medium','High'][i]} <b>${fmt(d)}</b></span>`).join('')}</div>`;
}

function render(r){
  const v=r.inputs;
  $('resultTitle').textContent=`Grip estimate: ${v.name}`;
  $('total').textContent=fmt(r.total_n,r.total_n<1?3:1);
  $('perFinger').textContent=fmt(r.per_finger_n,r.per_finger_n<1?3:1)+' N';
  $('status').className='status'+(['below_estimate','below_baseline'].includes(r.status)?' warning':'');
  $('status').textContent=r.status_text;
  if(v.limit_n!==null) $('status').textContent+=` Supplied limit: ${fmt(v.limit_n)} N total.`;
  $('diagramName').textContent=v.name;
  $('jaw1').textContent=fmt(r.per_finger_n,r.per_finger_n<1?3:1)+' N inward';
  $('jaw2').textContent=fmt(r.per_finger_n,r.per_finger_n<1?3:1)+' N inward';
  $('weight').textContent=force(r.weight_n)+' N';
  const cards=[
    ['01 / WEIGHT','Convert mass to load',`${fmt(v.mass_g/1000,3)} kg × 9.81 m/s²`,`${force(r.weight_n)} N`,'Gravity creates this downward load.'],
    ['02 / FRICTION','Find the nominal grip',`${force(r.weight_n)} N ÷ ${fmt(v.mu,3)}`,`${force(r.baseline_n)} N`,'Nominal total force from F = mg / µ, before the uncertainty allowance.'],
    ['03 / FUZZY LOGIC','Blend the allowance',`${fmt(v.mass_uncertainty_pct,1)}% mass / ${fmt(v.friction_uncertainty_pct,1)}% friction`,`${fmt(r.multiplier,3)}×`,`Adds ${fmt(r.margin_pct,1)}% to the nominal force under the assumed bounds.`],
    ['04 / ESTIMATE','Apply the multiplier',`${force(r.baseline_n)} N × ${fmt(r.multiplier,3)}`,`${force(r.total_n)} N`,'Total for both fingers. Divide by two for each equal contact.']
  ];
  $('trace').innerHTML=cards.map(([n,title,eq,value,note])=>`<div class="trace-card"><span class="n">${n}</span><h3>${title}</h3><div class="equation">${eq}</div><strong>${value}</strong><p>${note}</p></div>`).join('');
  $('massChart').innerHTML=chart(r.mass_memberships,v.mass_uncertainty_pct);
  $('frictionChart').innerHTML=chart(r.friction_memberships,v.friction_uncertainty_pct);
  const active=r.rules.filter(rule=>rule.activation>1e-10);
  $('rules').innerHTML=active.map(rule=>`<tr><td>${escapeHTML(rule.mass_label)}</td><td>${escapeHTML(rule.friction_label)}</td><td>${fmt(rule.consequent,4)}×</td><td>${fmt(rule.activation,4)}</td><td>${fmt(rule.contribution,4)}</td></tr>`).join('');
  const sum=r.rules.reduce((s,row)=>s+row.activation,0);
  $('fuzzyFormula').textContent=`Multiplier = (${active.map(rule=>`${fmt(rule.activation,4)} × ${fmt(rule.consequent,4)}`).join(' + ')}) ÷ ${fmt(sum,4)} = ${fmt(r.multiplier,4)}`;
  $('matrix').innerHTML='<table><thead><tr><th>Mass increase ↓ / Friction reduction →</th><th>Low · 0%</th><th>Medium · 25%</th><th>High · 50%</th></tr></thead><tbody>'+r.rule_matrix.map((row,i)=>`<tr><th>${['Low · 0%','Medium · 25%','High · 50%'][i]}</th>${row.map(n=>`<td>${fmt(n,4)}×</td>`).join('')}</tr>`).join('')+'</tbody></table>';
  $('boundComparison').textContent=`Direct uncertainty-bound calculation: ${fmt(r.exact_bound_n)} N. Fuzzy estimate: ${fmt(r.total_n)} N. This interpolation equals or slightly exceeds the direct bound; it is not evidence that fuzzy logic is more accurate. Calculations use unrounded values.`;
  const units={mass_g:'g',mass_uncertainty_pct:'%',friction_uncertainty_pct:'%',limit_n:'N total'};
  $('inputRecord').innerHTML=`<p><strong>Input provenance:</strong> ${escapeHTML(provenance)} Values below are the assumptions submitted for this calculation.</p><table><tbody>${Object.entries(v).map(([key,value])=>`<tr><th>${escapeHTML(fieldNames[key]||key)}</th><td>${escapeHTML(value===null?'Not supplied':String(value)+(units[key]?' '+units[key]:''))}</td></tr>`).join('')}</tbody></table>`;
  step(3);
}

$('objectForm').onsubmit=async e=>{
  e.preventDefault();
  const form=$('objectForm');
  for(const input of form.querySelectorAll('input')){
    if(!input.checkValidity() || (input.required&&!input.value.trim())){
      input.setAttribute('aria-invalid','true');input.focus();
      $('formError').textContent=`Check ${fieldNames[input.id].toLowerCase()}: ${input.validationMessage||'enter a value.'}`;return;
    }
  }
  const data={name:$('name').value.trim(),contact:$('contact').value.trim()};
  for(const key of numericKeys) data[key]=$(key).value===''?null:Number($(key).value);
  const button=form.querySelector('[type=submit]');button.disabled=true;
  try{
    const response=await fetch('/api/calculate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    const result=await response.json();
    if(!response.ok) throw new Error(result.error||'Calculation failed.');
    currentResult=result;render(result);
  }catch(error){$('formError').textContent=`${error.message} Check the values and that the local Python app is running.`;}
  finally{button.disabled=false;}
};

$('download').onclick=()=>{
  if(!currentResult)return;
  const record={method:'GripLab v2: static Coulomb-friction model + zero-order Sugeno uncertainty interpolation',provenance,created_at:new Date().toISOString(),...currentResult};
  const blob=new Blob([JSON.stringify(record,null,2)],{type:'application/json'});
  const url=URL.createObjectURL(blob);const link=document.createElement('a');link.href=url;link.download='griplab-calculation.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
loadProfiles();
