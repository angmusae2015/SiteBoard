"use strict";

const $ = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt = (n) => n.toLocaleString("ko-KR");
const pct = (n, total) => total ? (n / total * 100).toFixed(1) : "0.0";
const groupNames = {train:"학습", seen_patterns:"Seen", unseen_complex_holdout:"Unseen complex", unseen_rare:"Unseen rare"};
const groupColors = {train:"#6d9d87", seen_patterns:"#b4cbd7", unseen_complex_holdout:"#d9b98e", unseen_rare:"#bcaed2"};
const buckets = [{label:"1건", min:1, max:1}, {label:"2–5건", min:2, max:5}, {label:"6–10건", min:6, max:10}, {label:"11–20건", min:11, max:20}, {label:"21–50건", min:21, max:50}, {label:"51건+", min:51, max:Infinity}];
const state = {data:null, result:null, selected:null, tab:"patterns", bucket:null, timer:null, request:0, controller:null, visible:[], recordLimit:40};
const resultCache = new Map();
let curveSeries = null;

async function fetchJson(url, signal) {
  const response = await fetch(url, {signal});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function setStatus(text, busy = false) {
  $("compute-status").textContent = text;
  $("compute-status").classList.toggle("busy", busy);
}

function showError(error) {
  $("error").hidden = false;
  $("error").innerHTML = `${escapeHtml(error.message)} <button type="button" id="retry">다시 시도</button>`;
  $("retry").onclick = () => state.data ? requestThreshold(Number($("threshold").value)) : boot();
}

function summarize(clusters) {
  return {clusters:clusters.length, singletons:clusters.filter(c => c.count === 1).length, largest:clusters[0]?.count || 0, top10:clusters.slice(0, 10).reduce((sum,c) => sum+c.count,0) / state.data.records.length * 100};
}

function enrich(result) {
  if (result.enriched) return result;
  const baseline = state.data.baseline.clusters;
  for (const cluster of result.clusters) {
    const entries = Object.entries(cluster.baseline_overlap).sort((a,b) => b[1]-a[1]);
    cluster.bestBaseline = Number(entries[0][0]);
    cluster.change = cluster.count + baseline[cluster.bestBaseline].count - 2 * entries[0][1];
    cluster.indexSet = new Set(cluster.sample_indices);
    cluster.searchText = [cluster.rep_signature, ...cluster.sub_signatures].join(" ").toLowerCase();
  }
  result.enriched = true;
  return result;
}

async function requestThreshold(value) {
  const requestId = ++state.request;
  if (state.controller) state.controller.abort();
  state.controller = new AbortController();
  setStatus("계산 중", true);
  $("error").hidden = true;
  try {
    let result = resultCache.get(value);
    if (!result) {
      result = await fetchJson(`/api/clusters?threshold=${value.toFixed(3)}`, state.controller.signal);
      if (resultCache.size >= 40) resultCache.delete([...resultCache.keys()].find(k => k !== 0.85));
      resultCache.set(value, result);
    }
    if (requestId !== state.request) return;
    const old = state.result?.clusters[state.selected];
    const anchor = old?.sample_indices[0];
    state.result = enrich(result);
    state.selected = anchor === undefined ? 0 : result.clusters.find(c => c.indexSet.has(anchor))?.cluster_id ?? 0;
    state.recordLimit = 40;
    renderAll();
    setStatus(`적용 ${value.toFixed(3)}`);
  } catch (error) {
    if (error.name === "AbortError" || requestId !== state.request) return;
    setStatus("계산 실패");
    showError(error);
  }
}

function changeThreshold(raw, immediate = false, source = "slider") {
  const value = Math.round(Number(raw)*1000)/1000;
  if (!Number.isFinite(value) || value < 0 || value > 1) return;
  $("threshold").value = value;
  if (source !== "number" || immediate) $("threshold-number").value = value.toFixed(3);
  clearTimeout(state.timer);
  state.request++;
  state.controller?.abort();
  setStatus("계산 대기", true);
  if (immediate) requestThreshold(value);
  else state.timer = setTimeout(() => requestThreshold(value), 200);
}

function renderStats() {
  const current = summarize(state.result.clusters);
  const base = summarize(state.data.baseline.clusters);
  for (const key of ["clusters", "singletons", "largest", "top10"]) {
    $("stat-"+key).textContent = key === "top10" ? current[key].toFixed(1) : fmt(current[key]);
    const delta = current[key]-base[key];
    const rounded = key === "top10" ? Number(delta.toFixed(1)) : delta;
    const text = rounded === 0 ? "±0" : `${rounded > 0 ? "+" : "−"}${key === "top10" ? Math.abs(rounded).toFixed(1) : fmt(Math.abs(rounded))}`;
    $("delta-"+key).textContent = text + (key === "top10" ? " %p" : "");
    $("delta-"+key).classList.toggle("changed", rounded !== 0);
  }
  $("treemap-meta").textContent = `${fmt(state.result.total)}건`;
  $("treemap-count").textContent = fmt(current.clusters);
  $("applied-threshold").textContent = `θ ${state.result.threshold.toFixed(3)}`;
}

function treemapRects(clusters, width, height) {
  const scale = width*height/state.result.total;
  const items = clusters.map(c => ({cluster:c, area:c.count*scale}));
  const output = [];
  let x=0, y=0, w=width, h=height, cursor=0;
  const worst = (row, side) => {
    if (!row.length || side <= 0) return Infinity;
    const sum = row.reduce((s,i) => s+i.area,0);
    return Math.max(side*side*row[0].area/(sum*sum), sum*sum/(side*side*row[row.length-1].area));
  };
  while (cursor < items.length) {
    const row = [items[cursor++]];
    const side = Math.min(w,h);
    while (cursor < items.length && worst([...row,items[cursor]], side) <= worst(row,side)) row.push(items[cursor++]);
    const sum = row.reduce((s,i) => s+i.area,0);
    if (w >= h) {
      const strip = h > 0 ? sum/h : 0;
      let top=y;
      for (const item of row) { const cellHeight = strip > 0 ? item.area/strip : 0; output.push({...item,x,y:top,w:strip,h:cellHeight}); top+=cellHeight; }
      x+=strip; w=Math.max(0,w-strip);
    } else {
      const strip = w > 0 ? sum/w : 0;
      let left=x;
      for (const item of row) { const cellWidth = strip > 0 ? item.area/strip : 0; output.push({...item,x:left,y,w:cellWidth,h:strip}); left+=cellWidth; }
      y+=strip; h=Math.max(0,h-strip);
    }
  }
  return output;
}

function renderTreemap() {
  const node = $("treemap");
  const width = node.clientWidth;
  const height = node.clientHeight;
  if (!width || !height) return;
  const palette = ["#cbdce4","#d7e5e9","#b9cfd9","#dfe9ed","#c4d9dd","#e2e9e7","#d0dedf"];
  node.innerHTML = treemapRects(state.result.clusters,width,height).map(rect => {
    const c=rect.cluster;
    const color=palette[c.bestBaseline % palette.length];
    const text = rect.w>57 && rect.h>36;
    const label = `클러스터 ${c.cluster_id}, ${c.rep_signature}, ${c.count}건, ${pct(c.count,state.result.total)}%`;
    return `<button type="button" class="tree-cell${c.cluster_id===state.selected?" selected":""}" data-cluster="${c.cluster_id}" aria-label="${escapeHtml(label)}" aria-pressed="${c.cluster_id===state.selected}" style="left:${rect.x}px;top:${rect.y}px;width:${rect.w}px;height:${rect.h}px;--cell-color:${color}">${text?`<span class="cell-text"><span class="cell-name">${escapeHtml(c.rep_signature)}</span>${rect.h>62?`<span class="cell-count">${fmt(c.count)}<small>${pct(c.count,state.result.total)}%</small></span>`:""}</span>`:""}</button>`;
  }).join("");
  node.querySelectorAll(".tree-cell").forEach(button => {
    const c = state.result.clusters[Number(button.dataset.cluster)];
    button.onclick = () => selectCluster(c.cluster_id, true);
    button.onpointerenter = (event) => showTooltip(event,c);
    button.onpointermove = moveTooltip;
    button.onpointerleave = hideTooltip;
    button.onfocus = () => showTooltip({clientX:button.getBoundingClientRect().x, clientY:button.getBoundingClientRect().y},c);
    button.onblur = hideTooltip;
  });
  updateTreeHighlights();
}

function showTooltip(event,c) {
  $("tooltip").innerHTML = `<small>CLUSTER ${String(c.cluster_id).padStart(3,"0")}</small><div>${escapeHtml(c.rep_signature)}</div><b>${fmt(c.count)}</b>건 <small> / ${pct(c.count,state.result.total)}% / ${c.sub_signatures.length}개 패턴</small>`;
  $("tooltip").hidden = false;
  moveTooltip(event);
}

function moveTooltip(event) {
  const tooltip = $("tooltip");
  tooltip.style.left = `${Math.max(8,Math.min(event.clientX+14,window.innerWidth-tooltip.offsetWidth-12))}px`;
  tooltip.style.top = `${Math.max(8,Math.min(event.clientY+14,window.innerHeight-tooltip.offsetHeight-12))}px`;
}
function hideTooltip() { $("tooltip").hidden = true; }

function renderHistogram() {
  const counts = buckets.map(b => state.result.clusters.filter(c => c.count>=b.min && c.count<=b.max).length);
  const maximum = Math.max(...counts,1);
  $("histogram").innerHTML = buckets.map((bucket,i) => `<button type="button" class="hist-item${state.bucket===i?" selected":""}" aria-pressed="${state.bucket===i}" aria-label="${bucket.label}, ${counts[i]}개 클러스터" data-bucket="${i}"><span class="hist-number">${counts[i]}</span><span class="hist-bar" style="height:${counts[i]/maximum*68}px"></span><span class="hist-label">${bucket.label}</span></button>`).join("");
  $("histogram").querySelectorAll("button").forEach(button => button.onclick = () => {
    const index=Number(button.dataset.bucket);
    state.bucket=state.bucket===index?null:index;
    renderHistogram(); renderList(); renderDetail(); updateTreeHighlights();
  });
  $("clear-bucket").hidden=state.bucket===null;
}

function renderCurve() {
  const W=440, left=35, right=420, top=8, bottom=115;
  const current=state.result.clusters, baseline=state.data.baseline.clusters;
  const maxCount=Math.max(current.length,baseline.length);
  const x=(n)=>left+n/maxCount*(right-left);
  const y=(percent)=>bottom-percent/100*(bottom-top);
  const series=(clusters)=>{let sum=0;return [{n:0,p:0},...clusters.map((c,i)=>{sum+=c.count;return{n:i+1,p:sum/state.result.total*100};})];};
  const points=series(current), basePoints=series(baseline);
  const path=(data)=>data.map((point,i)=>`${i?"L":"M"}${x(point.n).toFixed(2)},${y(point.p).toFixed(2)}`).join(" ");
  const area=`${path(points)} L${x(current.length)},${bottom} L${left},${bottom} Z`;
  const grid=[0,50,100].map(p=>`<line x1="${left}" x2="${right}" y1="${y(p)}" y2="${y(p)}" stroke="#edf1f3"/><text x="${left-7}" y="${y(p)+3}" text-anchor="end" fill="#9ba9b1" font-size="9">${p}%</text>`).join("");
  const ticks=[0,Math.round(maxCount/2),maxCount].map(n=>`<text x="${x(n)}" y="132" text-anchor="middle" fill="#9ba9b1" font-size="9">${n}</text>`).join("");
  $("cumulative").innerHTML = `${grid}<path d="${area}" fill="#edf6f1"/><path d="${path(basePoints)}" fill="none" stroke="#b4c0c6" stroke-width="1.8" stroke-dasharray="4 4"/><path d="${path(points)}" fill="none" stroke="#378b70" stroke-width="2"/>${ticks}<text x="${W-2}" y="145" text-anchor="end" fill="#a4afb6" font-size="8">클러스터 수</text><g id="curve-hover" visibility="hidden"><line y1="${top}" y2="${bottom}" stroke="#7f9f93" stroke-dasharray="2 3"/><circle r="3" fill="#217e69"/></g>`;
  curveSeries={points,basePoints,maxCount,left,right,top,bottom};
  resetCurveReadout();
}

function resetCurveReadout() {
  if (!state.result) return;
  const top=Math.min(10,state.result.clusters.length);
  const count=state.result.clusters.slice(0,top).reduce((s,c)=>s+c.count,0);
  $("curve-readout").textContent=`상위 ${top}개 / ${pct(count,state.result.total)}%`;
}

function matchingClusters() {
  const query=$("search").value.trim().toLowerCase();
  const site=$("site-filter").value;
  const work=$("work-filter").value;
  const bucket=state.bucket===null?null:buckets[state.bucket];
  const filtered=state.result.clusters.filter(c => {
    if (bucket && (c.count<bucket.min || c.count>bucket.max)) return false;
    const patternHit=query && c.searchText.includes(query);
    return c.sample_indices.some(index=>{
      const r=state.data.records[index];
      if (site && r.site!==site) return false;
      if (work && !r.y.some(w=>w.workType===work)) return false;
      return !query || patternHit || `${r.x} ${r.site} ${r.image_id}`.toLowerCase().includes(query);
    });
  });
  const sort=$("sort").value;
  if (sort==="similarity") filtered.sort((a,b)=>a.min_similarity-b.min_similarity || b.count-a.count);
  if (sort==="change") filtered.sort((a,b)=>b.change-a.change || b.count-a.count);
  return filtered;
}

function renderList() {
  const clusters=matchingClusters();
  state.visible=clusters.map(c=>c.cluster_id);
  if (!state.visible.includes(state.selected)) state.selected=clusters[0]?.cluster_id ?? null;
  $("list-count").textContent=`${clusters.length} / ${state.result.clusters.length}`;
  $("cluster-list").innerHTML=clusters.length?clusters.map(c=>`<button type="button" class="cluster-row${c.cluster_id===state.selected?" selected":""}" data-cluster="${c.cluster_id}" aria-pressed="${c.cluster_id===state.selected}"><div class="row-content"><div class="row-id">CLUSTER ${String(c.cluster_id).padStart(3,"0")}${c.change?'<span class="row-change">소속 변경</span>':""}</div><div class="row-name" title="${escapeHtml(c.rep_signature)}">${escapeHtml(c.rep_signature)}</div></div><div class="row-numbers"><b>${fmt(c.count)}</b><small>${c.sub_signatures.length}개 패턴</small></div></button>`).join(""):'<div class="empty">검색 결과 없음</div>';
  $("cluster-list").querySelectorAll("button").forEach(button=>button.onclick=()=>selectCluster(Number(button.dataset.cluster)));
}

function clearFilters() {
  $("search").value=""; $("site-filter").value=""; $("work-filter").value=""; $("sort").value="size"; state.bucket=null;
}

function selectCluster(id, reveal=false) {
  hideTooltip();
  state.recordLimit=40;
  if (reveal && !state.visible.includes(id)) {clearFilters(); renderHistogram();}
  state.selected=id;
  renderList(); renderDetail(); updateTreeHighlights();
  $("detail-body").scrollTop=0;
  if (reveal) $("cluster-list").querySelector(`[data-cluster="${id}"]`)?.scrollIntoView({block:"nearest", behavior:"instant"});
}

function updateTreeHighlights() {
  const filtering=$("search").value || $("site-filter").value || $("work-filter").value || state.bucket!==null;
  const visible=new Set(state.visible);
  $("treemap").querySelectorAll("button").forEach(button=>{
    const id=Number(button.dataset.cluster);
    button.classList.toggle("selected",id===state.selected);
    button.classList.toggle("matched",Boolean(filtering)&&visible.has(id));
    button.setAttribute("aria-pressed",String(id===state.selected));
  });
}

function diffText(reference,text) {
  if (reference===text) return escapeHtml(text);
  if (reference.length*text.length>250000) return `<mark>${escapeHtml(text)}</mark>`;
  const m=reference.length,n=text.length;
  const table=Array.from({length:m+1},()=>new Uint16Array(n+1));
  for(let i=m-1;i>=0;i--) for(let j=n-1;j>=0;j--) table[i][j]=reference[i]===text[j]?table[i+1][j+1]+1:Math.max(table[i+1][j],table[i][j+1]);
  const pieces=[];
  const push=(type,char)=>{const prev=pieces[pieces.length-1];if(prev?.type===type)prev.text+=char;else pieces.push({type,text:char});};
  let i=0,j=0;
  while(i<m || j<n){if(i<m && j<n && reference[i]===text[j]){push("same",text[j++]);i++;}else if(j<n && (i===m || table[i][j+1]>=table[i+1][j]))push("add",text[j++]);else push("remove",reference[i++]);}
  return pieces.map(p=>p.type==="same"?escapeHtml(p.text):p.type==="add"?`<mark>${escapeHtml(p.text)}</mark>`:`<del>${escapeHtml(p.text)}</del>`).join("");
}

function recordCard(index) {
  const r=state.data.records[index];
  return `<article class="record-card"><div class="record-meta"><span class="record-site">${escapeHtml(r.site)}</span><span>사진 ${r.image_id}</span><span>#${index}</span><span class="split-tag ${r.split}">${groupNames[r.split]}</span></div><p class="record-text">${escapeHtml(r.x)}</p><div class="work-items">${r.y.map(w=>`<span class="work-item">${escapeHtml(w.workType)}${w.spec?` <b>${escapeHtml(w.spec)}</b>`:""}<b>× ${escapeHtml(w.quantity)}</b></span>`).join("")}</div></article>`;
}

function renderDetail() {
  const c=state.selected===null?null:state.result.clusters[state.selected];
  const body=$("detail-body");
  body.setAttribute("aria-labelledby",`tab-${state.tab}`);
  if(!c){$("detail-header").innerHTML="";body.innerHTML='<div class="empty">선택한 클러스터 없음</div>';return;}
  const sites=[...new Set(c.sample_indices.map(i=>state.data.records[i].site))].sort();
  $("detail-header").innerHTML=`<div class="detail-topline"><span class="cluster-badge">CLUSTER ${String(c.cluster_id).padStart(3,"0")}</span><span class="detail-ratio">전체의 ${pct(c.count,state.result.total)}%</span></div><h3 class="detail-title">${escapeHtml(c.rep_signature)}</h3><div class="detail-metrics"><span>레코드<b>${fmt(c.count)}</b></span><span>패턴<b>${c.sub_signatures.length}</b></span><span>최소 유사도<b>${c.min_similarity.toFixed(3)}</b></span><div class="site-pills">${sites.map(s=>`<span class="site-pill">${escapeHtml(s)}</span>`).join("")}</div></div>`;
  if(state.tab==="patterns") renderPatterns(c);
  else if(state.tab==="records") renderRecords(c);
  else if(state.tab==="compare") renderCompare(c);
  else renderSplit(c);
}

function renderPatterns(c) {
  $("detail-body").innerHTML=`<table class="detail-table"><thead><tr><th>정규화 패턴</th><th class="numeric">건수</th><th class="numeric">대표와 유사도</th></tr></thead><tbody>${c.patterns.map((p,index)=>`<tr><td><details data-pattern="${index}"><summary class="pattern-text">${diffText(c.rep_signature,p.text)}${index===0?'<span class="rep-tag">대표</span>':""}</summary><div class="pattern-records"></div></details></td><td class="numeric">${p.count}</td><td class="numeric"><span class="similarity-number">${p.similarity.toFixed(3)}</span><div class="similarity-track"><i style="width:${p.similarity*100}%"></i></div></td></tr>`).join("")}</tbody></table>`;
  $("detail-body").querySelectorAll("details").forEach(details=>details.addEventListener("toggle",()=>{
    if (!details.open || details.dataset.loaded) return;
    const sig=c.patterns[Number(details.dataset.pattern)].text;
    details.querySelector(".pattern-records").innerHTML=c.sample_indices.filter(i=>state.data.records[i].signature===sig).map(recordCard).join("");
    details.dataset.loaded="true";
  }));
}

function renderRecords(c) {
  const body=$("detail-body");
  const query=$("search").value.trim().toLowerCase(),site=$("site-filter").value,work=$("work-filter").value;
  const matches=(i)=>{const r=state.data.records[i];return(!site||r.site===site)&&(!work||r.y.some(w=>w.workType===work))&&(!query||`${r.x} ${r.site} ${r.image_id} ${r.signature}`.toLowerCase().includes(query));};
  const indices=[...c.sample_indices].sort((a,b)=>Number(matches(b))-Number(matches(a)) || a-b);
  body.innerHTML=indices.slice(0,state.recordLimit).map(recordCard).join("")+(indices.length>state.recordLimit?`<button id="more-records" class="text-button" type="button" style="display:block;margin:14px auto">더 보기 (${state.recordLimit} / ${indices.length})</button>`:"");
  $("more-records")?.addEventListener("click",()=>{const scroll=body.scrollTop;state.recordLimit+=40;renderRecords(c);body.scrollTop=scroll;});
}

function renderCompare(c) {
  const overlaps=Object.entries(c.baseline_overlap).sort((a,b)=>b[1]-a[1]);
  const baseline=state.data.baseline.clusters;
  let html=`<div class="compare-section"><div class="section-label"><span>현재 클러스터 ← 기준 0.85</span><span>${c.count}건</span></div>${overlaps.map(([id,count])=>`<div class="overlap-row"><div class="overlap-name"><span>기준 #${id} / ${baseline[id].count}건</span>${escapeHtml(baseline[id].rep_signature)}</div><div class="overlap-track"><i style="width:${count/c.count*100}%"></i></div><div class="overlap-number">${count}건</div></div>`).join("")}</div>`;
  html+=`<div class="compare-section"><div class="section-label"><span>기준 0.85 → 현재 클러스터</span><span>${overlaps.length}개 기준 클러스터</span></div>${overlaps.map(([id])=>{
    const destinations=state.result.clusters.filter(d=>d.baseline_overlap[id]).sort((a,b)=>b.baseline_overlap[id]-a.baseline_overlap[id]);
    return `<details${overlaps.length===1?" open":""}><summary>기준 #${id} / ${baseline[id].count}건 → ${destinations.length}개</summary>${destinations.map(d=>`<div class="destination"><button type="button" data-destination="${d.cluster_id}">#${d.cluster_id} ${escapeHtml(d.rep_signature)}</button><span>${d.baseline_overlap[id]}건</span></div>`).join("")}</details>`;
  }).join("")}</div>`;
  $("detail-body").innerHTML=html;
  $("detail-body").querySelectorAll("[data-destination]").forEach(button=>button.onclick=()=>selectCluster(Number(button.dataset.destination),true));
}

function renderSplit(c) {
  const sites={};
  const workTypes={};
  c.sample_indices.forEach(i=>{
    const record=state.data.records[i];
    sites[record.site]=(sites[record.site]||0)+1;
    new Set(record.y.map(w=>w.workType)).forEach(work=>{workTypes[work]=(workTypes[work]||0)+1;});
  });
  const splitRows=Object.keys(groupNames).map(group=>{
    const count=c.split_counts[group]||0;
    return `<div class="split-legend-row"><i style="background:${groupColors[group]}"></i><span>${groupNames[group]}</span><b>${count}건</b><span>${pct(count,c.count)}%</span></div>`;
  }).join("");
  const siteRows=Object.entries(sites).sort((a,b)=>b[1]-a[1]).map(([site,count])=>`<div class="site-bar-row"><span>${escapeHtml(site)}</span><div class="site-bar"><i style="width:${count/c.count*100}%"></i></div><b>${count}</b></div>`).join("");
  const workRows=Object.entries(workTypes).sort((a,b)=>b[1]-a[1]).map(([work,count])=>`<div class="site-bar-row work-type-row"><span>${escapeHtml(work)}</span><div class="site-bar"><i style="width:${count/c.count*100}%"></i></div><b>${count}</b></div>`).join("");
  $("detail-body").innerHTML=`<div class="split-summary"><div class="section-label"><span>09번 고정 분할</span><span>${c.count}건</span></div><div class="split-stack">${Object.keys(groupNames).map(g=>`<i style="width:${(c.split_counts[g]||0)/c.count*100}%;background:${groupColors[g]}" title="${groupNames[g]} ${c.split_counts[g]||0}건"></i>`).join("")}</div>${splitRows}<div class="site-section"><div class="section-label"><span>현장 분포</span><span>${Object.keys(sites).length}개 현장</span></div>${siteRows}</div><div class="site-section"><div class="section-label"><span>공종별 레코드</span><span>${Object.keys(workTypes).length}개 공종</span></div>${workRows}</div></div>`;
}

function renderAll() {
  renderStats(); renderHistogram(); renderCurve(); renderList(); renderTreemap(); renderDetail();
}

function bindEvents() {
  $("threshold").oninput=(event)=>changeThreshold(event.target.value);
  $("threshold").onchange=(event)=>changeThreshold(event.target.value,true);
  $("threshold-number").oninput=(event)=>{if(event.target.value!=="")changeThreshold(event.target.value,false,"number");};
  $("threshold-number").onchange=(event)=>{
    if(event.target.value==="" || !event.target.validity.valid){event.target.value=Number($("threshold").value).toFixed(3);return;}
    changeThreshold(event.target.value,true,"number");
  };
  $("reset").onclick=()=>changeThreshold(0.85,true);
  const filter=()=>{if(!state.result)return;state.recordLimit=40;renderList();renderDetail();updateTreeHighlights();};
  $("search").oninput=filter;
  ["site-filter","work-filter","sort"].forEach(id=>$(id).onchange=filter);
  $("clear-filters").onclick=()=>{clearFilters();renderHistogram();filter();};
  $("clear-bucket").onclick=()=>{state.bucket=null;renderHistogram();filter();};
  const tabs=[...document.querySelectorAll("[data-tab]")];
  tabs.forEach((button,index)=>{
    button.onclick=()=>{state.tab=button.dataset.tab;state.recordLimit=40;tabs.forEach(t=>{t.setAttribute("aria-selected",String(t===button));t.tabIndex=t===button?0:-1;});renderDetail();$("detail-body").scrollTop=0;};
    button.tabIndex=index===0?0:-1;
    button.onkeydown=(event)=>{
      let next;
      if(event.key==="ArrowRight")next=(index+1)%tabs.length;
      if(event.key==="ArrowLeft")next=(index+tabs.length-1)%tabs.length;
      if(event.key==="Home")next=0;
      if(event.key==="End")next=tabs.length-1;
      if(next!==undefined){event.preventDefault();tabs[next].focus();tabs[next].click();}
    };
  });
  $("cumulative").onpointermove=(event)=>{
    if(!curveSeries)return;
    const r=$("cumulative").getBoundingClientRect();
    const px=(event.clientX-r.left)/r.width*440;
    const s=curveSeries,n=Math.max(0,Math.min(s.maxCount,Math.round((px-s.left)/(s.right-s.left)*s.maxCount)));
    const current=s.points[Math.min(n,s.points.length-1)],base=s.basePoints[Math.min(n,s.basePoints.length-1)];
    const x=s.left+n/s.maxCount*(s.right-s.left),y=s.bottom-current.p/100*(s.bottom-s.top);
    const hover=$("curve-hover");hover.setAttribute("visibility","visible");
    const line=hover.querySelector("line");line.setAttribute("x1",x);line.setAttribute("x2",x);
    const circle=hover.querySelector("circle");circle.setAttribute("cx",x);circle.setAttribute("cy",y);
    $("curve-readout").textContent=`상위 ${n}개 / 현재 ${current.p.toFixed(1)}% / 기준 ${base.p.toFixed(1)}%`;
  };
  $("cumulative").onpointerleave=()=>{$("curve-hover")?.setAttribute("visibility","hidden");resetCurveReadout();};
  let resizeTimer;
  new ResizeObserver(()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{if(state.result)renderTreemap();},80);}).observe($("treemap"));
}

async function boot() {
  $("error").hidden=true;
  setStatus("불러오는 중",true);
  ["threshold","threshold-number","reset","search","site-filter","work-filter","sort","clear-filters"].forEach(id=>$(id).disabled=true);
  try {
    state.data=await fetchJson("/api/bootstrap");
    resultCache.set(0.85,state.data.baseline);
    state.result=enrich(state.data.baseline);state.selected=0;
    $("total-records").textContent=fmt(state.data.records.length);
    $("total-patterns").textContent=fmt(state.data.unique_patterns);
    $("site-filter").innerHTML='<option value="">전체 현장</option>'+state.data.sites.map(site=>`<option value="${escapeHtml(site)}">${escapeHtml(site)}</option>`).join("");
    $("work-filter").innerHTML='<option value="">전체 공종</option>'+state.data.work_types.map(work=>`<option value="${escapeHtml(work)}">${escapeHtml(work)}</option>`).join("");
    renderAll();setStatus("적용 0.850");
    ["threshold","threshold-number","reset","search","site-filter","work-filter","sort","clear-filters"].forEach(id=>$(id).disabled=false);
  } catch(error){setStatus("연결 실패");showError(error);}
}

bindEvents();
boot();
