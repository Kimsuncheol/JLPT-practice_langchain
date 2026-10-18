import json, html
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Preformatted, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
spec=json.load(open('tmp/pdfs/openapi.json'))
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='CodeSmall',fontName='Courier',fontSize=8,leading=10,backColor=colors.HexColor('#f0f4f8'),borderPadding=9))
styles['Title'].textColor=colors.HexColor('#12324a')
styles['Heading1'].textColor=colors.HexColor('#12324a')
styles['BodyText'].fontSize=8.5
styles['BodyText'].leading=12
story=[]
def p(t,style='BodyText'): return Paragraph(html.escape(str(t)),styles[style])
def add(t,style='BodyText'): story.extend([p(t,style),Spacer(1,5)])
def table(rows,widths):
 t=Table([[p(c) for c in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4edf4')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#9aafbf')),('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f7f9fb')])]))
 story.extend([t,Spacer(1,12)])
def code(obj):
 s=json.dumps(obj,indent=2,ensure_ascii=True) if not isinstance(obj,str) else obj
 story.extend([Preformatted(s,styles['CodeSmall'],maxLineLength=88),Spacer(1,12)])
add('JLPT Practice & Tutor','Title');add('API reference | Application v0.1.0 | 18 October 2026','Heading2')
add('Implementation reference for the current FastAPI + LangChain + Ollama project. Request and response fields below are derived from the application OpenAPI schema; behavioral notes are checked against routes, validators, services and configuration.')
add('Connection and conventions','Heading1')
add('Local base URL: http://127.0.0.1:8000. Deployment base URL: your deployed application origin. POST requests use Content-Type: application/json. Successful API responses use HTTP 200 and JSON. There is no application-level authentication declared in the current routes; OLLAMA_API_KEY is a backend-to-model proxy credential.')
add('Interactive reference: GET /docs; alternative reference: GET /redoc; machine-readable schema: GET /openapi.json. These are FastAPI default documentation endpoints.')
table([['Method / path','Use','Model call'],['GET /health','Lightweight application status','No'],['POST /api/practice/generate','Batch vocabulary, grammar or reading questions','Yes'],['POST /api/practice/evaluate','Deterministic answer comparison','No'],['POST /api/tutor','Tutoring for a selected practice item','Yes'],['POST /api/practice/question','Single question, optional topic','Yes'],['POST /api/tutor/chat','General or question-specific tutoring','Yes'],['GET /','HTML practice interface','No']],[220,230,55])
add('Integration and privacy','Heading1')
add('Use generate -> evaluate -> tutor for a complete practice flow. The bundled browser UI performs grading in browser memory and calls generate and tutor directly. The evaluate API is available to other clients. Send the answer key and explanations returned during generation when evaluating.')
add('Study and tutor state are ephemeral: no database or persistent conversation store. Clients resend bounded tutor history with each request. Practice, tutor and root responses carry Cache-Control: no-store and Pragma: no-cache, including errors covered by middleware. Health checks do not verify model connectivity. Generated content is unofficial, difficulty is approximate, and validation does not guarantee linguistic accuracy or predict official scores.')
add('Shared limits','Heading1')
add('Levels: N5, N4, N3, N2, N1 (case-sensitive). Explanation languages: English, Korean, Japanese; batch and tutor requests also normalize en/ko/ja and case variants. Batch categories normalize to lowercase. Batch count is 1-5, limited to 1-3 for reading; count is accepted as an alias for question_count. Tutor history permits at most 6 messages, each with role user or assistant and content 1-500 characters. Current request-body limit defaults to 4096 bytes; configuration allows 512-16384 bytes. Model timeout defaults to 45 seconds and is configurable from 1-120 seconds.')
endpoints=[('/health','get','HealthResponse'),('/api/practice/generate','post','PracticeGenerateResponse'),('/api/practice/evaluate','post','AnswerEvaluationResponse'),('/api/tutor','post','TutorResponse'),('/api/practice/question','post','PracticeQuestionResponse'),('/api/tutor/chat','post','TutorChatResponse')]
notes={
'/health':'Returns configured model name, tracing status and persistence status. base_url_configured only reports a nonempty configured URL; this is liveness/configuration information, not an Ollama readiness probe.',
'/api/practice/generate':'Alias: POST /practice/generate (hidden from OpenAPI). Produces original Japanese questions with four distinct choices, one correct index and four explanations. Reading requires a Japanese passage up to 800 characters and supporting sentences in each option explanation. Stable question IDs are derived from content. Unknown request fields are rejected.',
'/api/practice/evaluate':'Alias: POST /practice/evaluate. Trims selected answer and answer key before exact comparison. options plus correct_index take precedence over correct_answer. Without a usable key, returns is_correct=false and feedback stating no answer key was provided. With four options and four explanations, returns distractor feedback. Does not call Ollama. Client-supplied keys make this self-study grading, not tamper-proof scoring.',
'/api/tutor':'Alias: POST /tutor (hidden from OpenAPI). Requires selected question context (object or nonempty text up to 500 characters). question aliases: current_question, practice_item; message aliases: user_message, query; history aliases: recent_messages, messages. Unknown top-level fields are rejected. Selected context ignores extra fields and bounds option text to 200 and explanation text to 600 characters.',
'/api/practice/question':'Alias: POST /practice/question. Single-question contract differs from batch generation: supports kanji as well as vocabulary, grammar and reading, and accepts an optional topic. Returns question and explanation rather than batch question_text and option_explanations. No explanation_language request field.',
'/api/tutor/chat':'Alias: POST /tutor/chat. question is optional, allowing general tutoring. Accepts the same question aliases as /api/tutor. Its response lacks question_id and ai_generated. History is supplied by the client on every request.'}
examples={
'/api/practice/generate':{'level':'N5','category':'grammar','question_count':2,'explanation_language':'English'},
'/api/practice/evaluate':{'level':'N5','question':'Choose the topic particle.','selected_answer':'ha','options':['ha','wo','ni','de'],'correct_index':0,'option_explanations':['Marks the topic.','Marks an object.','Marks a destination.','Marks a location or means.']},
'/api/tutor':{'level':'N5','explanation_language':'English','question':{'id':'example-question','question_text':'Explain the topic particle.','options':['ha','wo','ni','de'],'correct_index':0},'history':[],'message':'Why does the first option fit?'},
'/api/practice/question':{'level':'N4','category':'kanji','topic':'daily life'},
'/api/tutor/chat':{'level':'N5','explanation_language':'English','message':'Explain the difference between topic and subject particles.','history':[]}}
def typename(s):
 if '$ref' in s:return s['$ref'].split('/')[-1]
 if 'anyOf' in s:return ' | '.join(typename(x) for x in s['anyOf'])
 if s.get('type')=='array':return 'list['+typename(s['items'])+']'
 if 'enum' in s:return ' / '.join(map(str,s['enum']))
 if 'const' in s:return str(s['const']).lower()
 return s.get('type','object')
for path,method,resp in endpoints:
 story.append(PageBreak());add(method.upper()+' '+path,'Heading1');add(notes[path])
 op=spec['paths'][path][method]
 if 'requestBody' in op:
  req=op['requestBody']['content']['application/json']['schema']['$ref'].split('/')[-1]
  add('Request: '+req,'Heading2');add('See the schema catalog for every field, default and constraint. Illustrative request:');code(examples[path])
  code('curl -X POST "http://127.0.0.1:8000'+path+'" \\\n  -H "Content-Type: application/json" \\\n  -d \''+json.dumps(examples[path],separators=(',',':'))+'\'') if path=='/api/practice/generate' else None
 add('Response: '+resp,'Heading2')
 schema=spec['components']['schemas'][resp]
 table([['Field','Type']] + [[k,typename(v)] for k,v in schema['properties'].items()],[225,280])
 if path=='/health':code({'status':'ok','model':'qwen2.5:7b','base_url_configured':True,'tracing_disabled':True,'persistence':'none'})
 add('Error behavior','Heading2')
 add('All POST endpoints can return 413 for oversized bodies and 422 for invalid requests. '+('Model routes also return 429 (quota/rate limit), 503 (unavailable model/service), and 502 (generation or processing failure). Timeout is explicitly mapped to 504 on batch and tutor routes; the single-question route can map timeout failures to 502.' if path not in ['/health','/api/practice/evaluate'] else 'No external model call is made by this endpoint.'))
story.append(PageBreak());add('Errors and client handling','Heading1')
table([['Status','Meaning / handling'],['200','Successful request. Inspect response fields for evaluation result.'],['413','Body exceeds configured byte limit. Reduce history/context or payload size.'],['422','Request/schema validation failed. Inspect detail entries and field locations.'],['429','Upstream quota or rate limit. Back off before retrying.'],['502','Upstream processing, structured output or validation failed.'],['503','Ollama service/model unavailable or configuration invalid.'],['504','Model request timed out on routes that map timeout explicitly.']],[65,440])
add('Non-validation errors generally use {"detail": "message"}. FastAPI validation errors use a detail array. No Retry-After header is explicitly set by these routes. Retrying a generation call may produce different questions.')
code({'detail':[{'type':'enum','loc':['body','level'],'msg':'Input should be a supported JLPT level','input':'N0'}]})
add('The 422 example above is illustrative; exact framework message wording can vary. The current Render guard rejects localhost/127.0.0.1 model URLs when RENDER=true. Use a network-reachable model endpoint for cloud deployment.')
add('Schema catalog','Heading1');add('The following catalog includes request models, response models and nested types exposed by OpenAPI. Required means the field must be present. Optional fields may still have defaults. Custom validation rules described earlier also apply. Descriptions and constraints follow the current source code.')
for name,s in sorted(spec['components']['schemas'].items()):
 if name in ['HTTPValidationError','ValidationError']:continue
 if 'enum' in s:
  add(name+': '+', '.join(s['enum']));continue
 story.append(PageBreak());add(name,'Heading1')
 if s.get('description'):add(s['description'])
 rows=[['Field / presence','Type / constraints / default']]
 for k,v in s.get('properties',{}).items():
  info=[typename(v)]
  for key in ['minimum','maximum','minLength','maxLength','minItems','maxItems']:
   if key in v:info.append(key+'='+str(v[key]))
  if 'default' in v:info.append('default='+json.dumps(v['default'],ensure_ascii=True))
  if v.get('description'):info.append(v['description'])
  rows.append([k+(' (required)' if k in s.get('required',[]) else ' (optional)'),'; '.join(info)])
 table(rows,[180,325])
add('Source and scope','Heading2');add('Source files: app/routes.py, app/schemas.py, app/services.py, app/config.py, app/main.py and app/static/app.js. This reference documents the local checkout on 18 October 2026. Examples are illustrative and do not represent live model output. No running Ollama instance was required to build this reference.')
def footer(c,d):
 c.setStrokeColor(colors.HexColor('#cad6df'));c.line(45,43,550,43);c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#53697a'));c.drawString(45,30,'JLPT Practice & Tutor | API reference');c.drawRightString(550,30,str(d.page))
out='output/pdf/jlpt-practice-api-reference.pdf'
SimpleDocTemplate(out,pagesize=(595.28,841.89),rightMargin=45,leftMargin=45,topMargin=45,bottomMargin=58,title='JLPT Practice & Tutor - API Reference',author='JLPT Practice Project').build(story,onFirstPage=footer,onLaterPages=footer)
print(out)
