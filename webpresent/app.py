from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import json
import uuid
import time
import os
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading
import asyncio

# Импорт нашего ассистента и системы пакетного обучения
try:
   from neurograph.integration import create_default_engine
   NEUROGRAPH_AVAILABLE = True
except ImportError:
   NEUROGRAPH_AVAILABLE = False
   print("⚠️ NeuroGraph не найден, работаем в демо-режиме")

# Импортируем систему пакетного обучения
try:
   from bulk_learning import BulkLearningSystem, TextChunker
   BULK_LEARNING_AVAILABLE = True
except ImportError:
   BULK_LEARNING_AVAILABLE = False
   print("⚠️ Система пакетного обучения недоступна")

app = Flask(__name__)
CORS(app)

class WebNeuroGraphAssistant:
   """Веб-версия персонального ассистента NeuroGraph."""
   
   def __init__(self):
       self.user_sessions = {}  # Хранение сессий пользователей
       self.global_stats = {
           "total_users": 0,
           "total_messages": 0,
           "start_time": datetime.now(),
           "system_healthy": NEUROGRAPH_AVAILABLE
       }
       
       # Инициализация NeuroGraph
       if NEUROGRAPH_AVAILABLE:
           try:
               self.engine = create_default_engine()
               self.memory = self.engine.provider.get_component('memory')
               self.graph = self.engine.provider.get_component('semgraph')
               
               # Инициализация системы пакетного обучения
               if BULK_LEARNING_AVAILABLE:
                   self.bulk_system = BulkLearningSystem(
                       neurograph_engine=self.engine,
                       batch_size=5,
                       delay_between_batches=0.1,
                       progress_callback=self._bulk_progress_callback
                   )
               else:
                   self.bulk_system = None
               
               print("✅ NeuroGraph успешно инициализирован")
           except Exception as e:
               print(f"❌ Ошибка инициализации NeuroGraph: {e}")
               self.engine = None
               self.memory = None
               self.graph = None
               self.bulk_system = None
               self.global_stats["system_healthy"] = False
       else:
           self.engine = None
           self.memory = None
           self.graph = None
           self.bulk_system = None
       
       # Для отслеживания прогресса пакетного обучения
       self.bulk_progress = {}
       self.bulk_tasks = {}  # Отслеживание активных задач
   
   def _bulk_progress_callback(self, message: str):
       """Callback для отслеживания прогресса пакетного обучения."""
       timestamp = datetime.now().isoformat()
       
       # Добавляем в общий лог прогресса
       if 'bulk_learning' not in self.bulk_progress:
           self.bulk_progress['bulk_learning'] = []
       
       self.bulk_progress['bulk_learning'].append({
           'timestamp': timestamp,
           'message': message
       })
       
       # Ограничиваем размер лога
       if len(self.bulk_progress['bulk_learning']) > 100:
           self.bulk_progress['bulk_learning'] = self.bulk_progress['bulk_learning'][-50:]
       
       print(f"[BULK] {message}")
   
   def process_bulk_text(self, text: str, session_id: str, 
                        chunk_size: int = 500, 
                        importance: float = 0.7,
                        category: str = "bulk_text") -> Dict:
       """Обрабатывает большой текст через систему пакетного обучения."""
       
       if not self.bulk_system:
           return {
               "success": False,
               "error": "Система пакетного обучения недоступна"
           }
       
       task_id = str(uuid.uuid4())
       
       try:
           # Настраиваем chunker
           self.bulk_system.chunker.chunk_size = chunk_size
           
           # Подготавливаем метаданные
           metadata = {
               'session_id': session_id,
               'category': category,
               'task_id': task_id,
               'text_length': len(text)
           }
           
           # Запускаем обработку
           self.bulk_tasks[task_id] = {
               'start_time': datetime.now(),
               'status': 'processing',
               'session_id': session_id
           }
           
           # Обрабатываем текст
           result = self.bulk_system.process_large_text(
               text=text,
               metadata=metadata,
               importance=importance
           )
           
           # Обновляем статус задачи
           self.bulk_tasks[task_id].update({
               'status': 'completed' if result['success'] else 'failed',
               'end_time': datetime.now(),
               'result': result
           })
           
           # Обновляем локальную базу сессии
           session = self.get_or_create_session(session_id)
           session["total_facts"] += result.get('processed_chunks', 0)
           
           return {
               "success": True,
               "task_id": task_id,
               "result": result
           }
           
       except Exception as e:
           # Обновляем статус задачи при ошибке
           if task_id in self.bulk_tasks:
               self.bulk_tasks[task_id].update({
                   'status': 'failed',
                   'end_time': datetime.now(),
                   'error': str(e)
               })
           
           return {
               "success": False,
               "task_id": task_id,
               "error": str(e)
           }
   
   def process_bulk_file(self, file_content: str, filename: str, 
                        session_id: str, **kwargs) -> Dict:
       """Обрабатывает загруженный файл."""
       
       if not self.bulk_system:
           return {
               "success": False,
               "error": "Система пакетного обучения недоступна"
           }
       
       task_id = str(uuid.uuid4())
       
       try:
           # Определяем категорию по расширению файла
           file_ext = os.path.splitext(filename)[1].lower()
           category_map = {
               '.txt': 'text_document',
               '.md': 'markdown',
               '.py': 'code_python',
               '.js': 'code_javascript',
               '.html': 'html_document',
               '.json': 'json_data'
           }
           category = category_map.get(file_ext, 'document')
           
           # Подготавливаем метаданные
           metadata = {
               'filename': filename,
               'file_extension': file_ext,
               'session_id': session_id,
               'task_id': task_id,
               'upload_time': datetime.now().isoformat()
           }
           
           # Запускаем обработку
           self.bulk_tasks[task_id] = {
               'start_time': datetime.now(),
               'status': 'processing',
               'session_id': session_id,
               'filename': filename
           }
           
           result = self.bulk_system.process_large_text(
               text=file_content,
               metadata=metadata,
               importance=kwargs.get('importance', 0.7)
           )
           
           # Обновляем статус задачи
           self.bulk_tasks[task_id].update({
               'status': 'completed' if result['success'] else 'failed',
               'end_time': datetime.now(),
               'result': result
           })
           
           # Обновляем локальную базу сессии
           session = self.get_or_create_session(session_id)
           session["total_facts"] += result.get('processed_chunks', 0)
           
           return {
               "success": True,
               "task_id": task_id,
               "filename": filename,
               "result": result
           }
           
       except Exception as e:
           if task_id in self.bulk_tasks:
               self.bulk_tasks[task_id].update({
                   'status': 'failed',
                   'end_time': datetime.now(),
                   'error': str(e)
               })
           
           return {
               "success": False,
               "task_id": task_id,
               "filename": filename,
               "error": str(e)
           }
   
   def get_bulk_task_status(self, task_id: str) -> Dict:
       """Получает статус задачи пакетного обучения."""
       
       if task_id not in self.bulk_tasks:
           return {
               "success": False,
               "error": "Задача не найдена"
           }
       
       task_info = self.bulk_tasks[task_id].copy()
       
       # Добавляем дополнительную информацию
       if 'start_time' in task_info:
           task_info['start_time'] = task_info['start_time'].isoformat()
       if 'end_time' in task_info:
           task_info['end_time'] = task_info['end_time'].isoformat()
           
           # Вычисляем длительность
           start = self.bulk_tasks[task_id]['start_time']
           end = self.bulk_tasks[task_id]['end_time']
           duration = (end - start).total_seconds()
           task_info['duration_seconds'] = duration
       
       return {
           "success": True,
           "task": task_info
       }
   
   def get_bulk_progress_log(self, limit: int = 50) -> List[Dict]:
       """Получает лог прогресса пакетного обучения."""
       
       if 'bulk_learning' not in self.bulk_progress:
           return []
       
       return self.bulk_progress['bulk_learning'][-limit:]
   
   def cleanup_old_tasks(self, max_age_hours: int = 24):
       """Очищает старые задачи."""
       
       cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
       tasks_to_remove = []
       
       for task_id, task_info in self.bulk_tasks.items():
           if task_info.get('end_time') and task_info['end_time'] < cutoff_time:
               tasks_to_remove.append(task_id)
       
       for task_id in tasks_to_remove:
           del self.bulk_tasks[task_id]
       
       return len(tasks_to_remove)
   
   def get_or_create_session(self, session_id: str = None) -> Dict:
       """Получает или создает пользовательскую сессию."""
       if not session_id:
           session_id = str(uuid.uuid4())
       
       if session_id not in self.user_sessions:
           self.user_sessions[session_id] = {
               "id": session_id,
               "created_at": datetime.now(),
               "messages": [],
               "local_knowledge": {},
               "user_name": f"Пользователь_{len(self.user_sessions) + 1}",
               "total_queries": 0,
               "total_facts": 0
           }
           self.global_stats["total_users"] += 1
           
           # Добавляем приветственное сообщение
           welcome_msg = {
               "id": str(uuid.uuid4()),
               "sender": "assistant",
               "content": f"Привет! Я NeuroGraph Assistant - ваш персональный ИИ-помощник с биоморфной архитектурой памяти. Я могу обучаться, запоминать информацию и помогать решать задачи. О чем хотите поговорить?",
               "timestamp": datetime.now().isoformat(),
               "confidence": 0.95,
               "sources": ["system"],
               "processing_time": 0
           }
           self.user_sessions[session_id]["messages"].append(welcome_msg)
       
       return self.user_sessions[session_id]
   
   async def process_message(self, session_id: str, message: str) -> Dict:
       """Обрабатывает сообщение пользователя."""
       session = self.get_or_create_session(session_id)
       start_time = time.time()
       
       # Добавляем сообщение пользователя в историю
       user_msg = {
           "id": str(uuid.uuid4()),
           "sender": "user",
           "content": message,
           "timestamp": datetime.now().isoformat(),
           "confidence": 1.0
       }
       session["messages"].append(user_msg)
       session["total_queries"] += 1
       self.global_stats["total_messages"] += 1
       
       try:
           # Обработка через NeuroGraph или fallback
           if self.engine:
               response = await self._process_with_neurograph(message, session)
           else:
               response = self._process_with_fallback(message, session)
           
           processing_time = (time.time() - start_time) * 1000
           
           # Создаем ответное сообщение
           assistant_msg = {
               "id": str(uuid.uuid4()),
               "sender": "assistant",
               "content": response["answer"],
               "timestamp": datetime.now().isoformat(),
               "confidence": response.get("confidence", 0.5),
               "sources": response.get("sources", []),
               "processing_time": processing_time,
               "meta": response.get("meta", {})
           }
           
           session["messages"].append(assistant_msg)
           
           return {
               "success": True,
               "message": assistant_msg,
               "session_stats": self._get_session_stats(session)
           }
           
       except Exception as e:
           error_msg = {
               "id": str(uuid.uuid4()),
               "sender": "assistant",
               "content": f"Извините, произошла ошибка при обработке вашего сообщения: {str(e)}",
               "timestamp": datetime.now().isoformat(),
               "confidence": 0.1,
               "sources": ["error_handler"],
               "processing_time": (time.time() - start_time) * 1000,
               "error": True
           }
           
           session["messages"].append(error_msg)
           
           return {
               "success": False,
               "message": error_msg,
               "error": str(e)
           }
   
   async def _process_with_neurograph(self, message: str, session: Dict) -> Dict:
       """Обработка через NeuroGraph."""
       try:
           # Симуляция async для демонстрации
           await self._async_delay(0.1)
           
           # Обработка специальных команд
           if any(cmd in message.lower() for cmd in ['запомни', 'обучи', 'сохрани']):
               return await self._handle_learning_command(message, session)
           
           if any(cmd in message.lower() for cmd in ['покажи память', 'что помнишь', 'статистика памяти']):
               return await self._handle_memory_command(session)
           
           if any(cmd in message.lower() for cmd in ['статус', 'система', 'диагностика']):
               return await self._handle_system_command()
           
           # Основной запрос к NeuroGraph
           response = self.engine.query(message)
           
           # Проверяем качество ответа и используем fallback если нужно
           if (hasattr(response, 'confidence') and response.confidence < 0.3) or \
              "не нашел" in response.primary_response.lower():
               
               fallback_response = self._search_local_knowledge(message, session)
               if fallback_response["confidence"] > response.confidence:
                   return {
                       "answer": f"{fallback_response['answer']}\n\n💡 *Дополнительно от NeuroGraph: {response.primary_response}*",
                       "confidence": max(fallback_response["confidence"], response.confidence),
                       "sources": ["локальная база", "NeuroGraph"],
                       "meta": {"combined_response": True}
                   }
           
           return {
               "answer": response.primary_response,
               "confidence": getattr(response, 'confidence', 0.7),
               "sources": getattr(response, 'sources', ["NeuroGraph"]),
               "meta": {
                   "neurograph_response": True,
                   "processing_time": getattr(response, 'processing_time', 0)
               }
           }
           
       except Exception as e:
           print(f"Ошибка NeuroGraph: {e}")
           return self._process_with_fallback(message, session)
   
   async def _handle_learning_command(self, message: str, session: Dict) -> Dict:
       """Обработка команды обучения."""
       # Извлекаем контент для обучения
       content = message
       for trigger in ['запомни', 'обучи', 'сохрани']:
           content = content.replace(trigger, '').strip()
       
       if not content:
           return {
               "answer": "Что именно вы хотите, чтобы я запомнил?",
               "confidence": 0.9,
               "sources": ["learning_system"]
           }
       
       # Обучаем NeuroGraph
       if self.engine:
           response = self.engine.learn(content)
           if response.success:
               # Сохраняем в локальную базу сессии
               fact_id = str(uuid.uuid4())
               session["local_knowledge"][fact_id] = {
                   "content": content,
                   "timestamp": datetime.now().isoformat(),
                   "category": "user_taught"
               }
               session["total_facts"] += 1
               
               return {
                   "answer": f"✅ Отлично! Я запомнил: '{content}'\n\nТеперь эта информация сохранена в моей памяти и я смогу использовать её в будущих разговорах.",
                   "confidence": 0.95,
                   "sources": ["learning_system", "NeuroGraph"],
                   "meta": {"learned_content": content}
               }
       
       # Fallback: сохраняем только локально
       fact_id = str(uuid.uuid4())
       session["local_knowledge"][fact_id] = {
           "content": content,
           "timestamp": datetime.now().isoformat(),
           "category": "user_taught"
       }
       session["total_facts"] += 1
       
       return {
           "answer": f"📝 Записал в локальную память: '{content}'\n\nБуду помнить это во время нашего разговора!",
           "confidence": 0.8,
           "sources": ["local_memory"]
       }
   
   async def _handle_memory_command(self, session: Dict) -> Dict:
       """Обработка команды показа памяти."""
       memory_info = []
       
       # Локальные знания сессии
       local_count = len(session["local_knowledge"])
       if local_count > 0:
           memory_info.append(f"📝 **Локальная память сессии**: {local_count} фактов")
           
           recent_facts = list(session["local_knowledge"].values())[-3:]
           for fact in recent_facts:
               memory_info.append(f"  • {fact['content']}")
       
       # NeuroGraph память
       if self.memory:
           try:
               stats = self.memory.get_memory_statistics()
               stm_size = stats['memory_levels']['stm']['size']
               ltm_size = stats['memory_levels']['ltm']['size']
               
               memory_info.append(f"\n🧠 **NeuroGraph память**:")
               memory_info.append(f"  • Кратковременная: {stm_size} элементов")
               memory_info.append(f"  • Долговременная: {ltm_size} элементов")
               
               # Пробуем показать примеры из памяти
               try:
                   recent_items = self.memory.get_recent_items()
                   if recent_items:
                       memory_info.append(f"\n📋 **Последние элементы STM**:")
                       for item in recent_items[:3]:
                           content = getattr(item, 'content', str(item))
                           if len(content) > 50:
                               content = content[:50] + "..."
                           memory_info.append(f"  • {content}")
               except Exception as e:
                   memory_info.append(f"\n⚠️ Ошибка получения элементов памяти: {e}")
               
           except Exception as e:
               memory_info.append(f"\n⚠️ Ошибка доступа к NeuroGraph памяти: {e}")
       
       if not memory_info:
           return {
               "answer": "Моя память пока пуста. Расскажите мне что-нибудь, и я это запомню!",
               "confidence": 0.9,
               "sources": ["memory_system"]
           }
       
       return {
           "answer": "🧠 **Состояние моей памяти:**\n\n" + "\n".join(memory_info) + "\n\n💡 *Переключитесь на вкладку 'Знания' для подробного просмотра!*",
           "confidence": 0.95,
           "sources": ["memory_system"]
       }
   
   async def _handle_system_command(self) -> Dict:
       """Обработка команды системной диагностики."""
       uptime = datetime.now() - self.global_stats["start_time"]
       uptime_str = str(uptime).split('.')[0]  # Убираем микросекунды
       
       status_info = [
           f"🖥️ **Системная диагностика:**\n",
           f"⏱️ **Время работы**: {uptime_str}",
           f"👥 **Активных пользователей**: {len(self.user_sessions)}",
           f"💬 **Всего сообщений**: {self.global_stats['total_messages']}",
           f"🧠 **NeuroGraph**: {'✅ Активен' if self.engine else '❌ Недоступен'}",
       ]
       
       if self.memory:
           try:
               stats = self.memory.get_memory_statistics()
               total_items = stats.get('total_items', 0)
               status_info.append(f"📊 **Элементов в памяти**: {total_items}")
           except:
               pass
       
       status_info.append(f"\n💡 *Переключитесь на вкладку 'Система' для детальной диагностики!*")
       
       return {
           "answer": "\n".join(status_info),
           "confidence": 0.99,
           "sources": ["system_monitor"]
       }
   
   def _process_with_fallback(self, message: str, session: Dict) -> Dict:
       """Fallback обработка без NeuroGraph."""
       # Поиск в локальных знаниях
       local_response = self._search_local_knowledge(message, session)
       if local_response["confidence"] > 0.5:
           return local_response
       
       # Предустановленные ответы
       responses = {
           "привет": "Привет! Рад вас видеть! Как дела?",
           "как дела": "У меня все отлично! Правда, сейчас работаю в упрощенном режиме, так как NeuroGraph недоступен. Но я все равно могу помочь!",
           "что ты умеешь": "Я умею запоминать информацию, отвечать на вопросы и поддерживать беседу. В полном режиме с NeuroGraph у меня гораздо больше возможностей!",
           "помоги": "Конечно помогу! Расскажите, что вас интересует, или попросите меня что-то запомнить.",
           "спасибо": "Пожалуйста! Всегда рад помочь! 😊"
       }
       
       message_lower = message.lower()
       for key, response in responses.items():
           if key in message_lower:
               return {
                   "answer": response,
                   "confidence": 0.8,
                   "sources": ["предустановленные ответы"]
               }
       
       # Общий ответ
       general_responses = [
           "Интересно! Расскажите больше об этом.",
           "Это требует размышлений. Можете дать больше контекста?",
           "Хм, новая для меня тема. Давайте изучим её вместе!",
           "Я пока думаю над этим. А что вы сами об этом знаете?"
       ]
       
       import random
       return {
           "answer": random.choice(general_responses),
           "confidence": 0.4,
           "sources": ["общие ответы"]
       }
   
   def _search_local_knowledge(self, query: str, session: Dict) -> Dict:
       """Поиск в локальных знаниях сессии."""
       if not session["local_knowledge"]:
           return {"answer": "", "confidence": 0.0, "sources": []}
       
       query_words = set(query.lower().split())
       relevant_facts = []
       
       for fact_data in session["local_knowledge"].values():
           content = fact_data["content"].lower()
           content_words = set(content.split())
           
           # Подсчет пересечений
           overlap = len(query_words.intersection(content_words))
           if overlap > 0:
               relevant_facts.append({
                   "content": fact_data["content"],
                   "relevance": overlap,
                   "timestamp": fact_data["timestamp"]
               })
       
       if not relevant_facts:
           return {"answer": "", "confidence": 0.0, "sources": []}
       
       # Сортируем по релевантности
       relevant_facts.sort(key=lambda x: x["relevance"], reverse=True)
       best_fact = relevant_facts[0]
       
       return {
           "answer": f"Из того, что я помню: {best_fact['content']}",
           "confidence": min(0.8, best_fact["relevance"] / 3),
           "sources": ["локальная память сессии"]
       }
   
   def _get_session_stats(self, session: Dict) -> Dict:
       """Получение статистики сессии."""
       return {
           "session_id": session["id"],
           "messages_count": len(session["messages"]),
           "local_facts": len(session["local_knowledge"]),
           "total_queries": session["total_queries"],
           "total_facts": session["total_facts"],
           "session_duration": str(datetime.now() - session["created_at"]).split('.')[0]
       }
   
   def get_session_knowledge(self, session_id: str) -> Dict:
       """Получение знаний сессии для вкладки 'Знания'."""
       session = self.get_or_create_session(session_id)
       
       knowledge_data = {
           "local_knowledge": session["local_knowledge"],
           "stm_items": [],
           "ltm_items": [],
           "stats": {
               "total_knowledge": len(session["local_knowledge"]),
               "active_memories": 0,
               "graph_nodes": 0,
               "connections": 0
           }
       }
       
       # Добавляем данные NeuroGraph если доступен
       if self.memory:
           try:
               stats = self.memory.get_memory_statistics()
               knowledge_data["stats"].update({
                   "active_memories": stats['memory_levels']['stm']['size'],
                   "graph_nodes": stats['memory_levels']['ltm']['size'],
                   "total_knowledge": stats.get('total_items', len(session["local_knowledge"]))
               })
               
               # Получаем последние элементы из памяти
               try:
                   recent_items = self.memory.get_recent_items()  # Убираем параметр limit
                   knowledge_data["stm_items"] = [
                       {
                           "content": getattr(item, 'content', str(item))[:100] + ("..." if len(str(getattr(item, 'content', str(item)))) > 100 else ""),
                           "tags": getattr(item, 'metadata', {}).get('tags', ['память'])
                       }
                       for item in (recent_items[:5] if recent_items else [])  # Ограничиваем до 5 элементов
                   ]
               except Exception as e:
                   print(f"Ошибка получения recent_items: {e}")
                   knowledge_data["stm_items"] = []
               
               # Пробуем получить данные из LTM
               try:
                   # Получаем доступ к LTM компоненту
                   if hasattr(self.memory, 'ltm') and self.memory.ltm:
                       ltm_items = []
                       # Пробуем получить элементы из LTM
                       if hasattr(self.memory.ltm, 'get_all_items'):
                           ltm_all = self.memory.ltm.get_all_items()
                           ltm_items = ltm_all[:5] if ltm_all else []
                       elif hasattr(self.memory.ltm, 'items'):
                           ltm_items = list(self.memory.ltm.items())[:5]
                       
                       knowledge_data["ltm_items"] = [
                           {
                               "content": getattr(item, 'content', str(item))[:100] + ("..." if len(str(getattr(item, 'content', str(item)))) > 100 else ""),
                               "tags": getattr(item, 'metadata', {}).get('tags', ['долговременная'])
                           }
                           for item in ltm_items
                       ]
                   else:
                       knowledge_data["ltm_items"] = []
                       
               except Exception as e:
                   print(f"Ошибка получения LTM данных: {e}")
                   knowledge_data["ltm_items"] = []
               
           except Exception as e:
               print(f"Ошибка получения данных памяти: {e}")
               knowledge_data["stats"]["error"] = str(e)
       
       return knowledge_data
   
   def get_system_status(self) -> Dict:
       """Получение системного статуса для вкладки 'Система'."""
       uptime = datetime.now() - self.global_stats["start_time"]
       
       status = {
           "components": {
               "nlp": {"status": "active" if self.engine else "inactive", "performance": 85},
               "semgraph": {"status": "active" if self.graph else "inactive", "performance": 92},
               "memory": {"status": "active" if self.memory else "inactive", "performance": 78},
               "processor": {"status": "active" if self.engine else "inactive", "performance": 88}
           },
           "stats": {
               "uptime": str(uptime).split('.')[0],
               "total_users": self.global_stats["total_users"],
               "total_messages": self.global_stats["total_messages"],
               "avg_response_time": "0.8s",
               "system_health": "healthy" if self.global_stats["system_healthy"] else "degraded"
           },
           "memory_stats": {}
       }
       
       if self.memory:
           try:
               memory_stats = self.memory.get_memory_statistics()
               status["memory_stats"] = memory_stats
           except Exception as e:
               status["memory_stats"] = {"error": str(e)}
       
       return status
   
   async def _async_delay(self, seconds: float):
       """Асинхронная задержка для симуляции обработки."""
       import asyncio
       await asyncio.sleep(seconds)

# Создаем глобальный экземпляр ассистента
assistant = WebNeuroGraphAssistant()

@app.route('/')
def index():
   """Главная страница."""
   return render_template('neurograph_assistant.html')

@app.route('/api/message', methods=['POST'])
def handle_message():
   """Обработка сообщения от пользователя."""
   try:
       data = request.get_json()
       message = data.get('message', '').strip()
       session_id = data.get('session_id', str(uuid.uuid4()))
       
       if not message:
           return jsonify({"error": "Пустое сообщение"}), 400
       
       # Обработка сообщения (синхронно для простоты)
       import asyncio
       loop = asyncio.new_event_loop()
       asyncio.set_event_loop(loop)
       result = loop.run_until_complete(assistant.process_message(session_id, message))
       loop.close()
       
       return jsonify(result)
       
   except Exception as e:
       return jsonify({"error": str(e)}), 500

@app.route('/api/bulk/text', methods=['POST'])
def bulk_learn_text():
   """Пакетное обучение на большом тексте."""
   try:
       data = request.get_json()
       text = data.get('text', '').strip()
       session_id = data.get('session_id', str(uuid.uuid4()))
       
       if not text:
           return jsonify({"error": "Текст не предоставлен"}), 400
       
       if len(text) < 100:
           return jsonify({"error": "Текст слишком короткий для пакетного обучения (минимум 100 символов)"}), 400
       
       # Параметры обработки
       chunk_size = data.get('chunk_size', 500)
       importance = data.get('importance', 0.7)
       category = data.get('category', 'bulk_text')
       
       # Запускаем пакетное обучение
       result = assistant.process_bulk_text(
           text=text,
           session_id=session_id,
           chunk_size=chunk_size,
           importance=importance,
           category=category
       )
       
       return jsonify(result)
       
   except Exception as e:
       return jsonify({"error": str(e)}), 500

@app.route('/api/bulk/file', methods=['POST'])
def bulk_learn_file():
   """Пакетное обучение на загруженном файле."""
   try:
       if 'file' not in request.files:
           return jsonify({"error": "Файл не предоставлен"}), 400
       
       file = request.files['file']
       if file.filename == '':
           return jsonify({"error": "Файл не выбран"}), 400
       
       # Проверяем тип файла
       allowed_extensions = {'.txt', '.md', '.py', '.js', '.html', '.json', '.csv'}
       file_ext = os.path.splitext(file.filename)[1].lower()
       
       if file_ext not in allowed_extensions:
           return jsonify({"error": f"Неподдерживаемый тип файла: {file_ext}"}), 400
       
       # Читаем содержимое файла
       try:
           if file_ext == '.csv':
               file_content = file.read().decode('utf-8')
           else:
               file_content = file.read().decode('utf-8')
       except UnicodeDecodeError:
           try:
               file_content = file.read().decode('cp1251')  # Попробуем windows-1251
           except UnicodeDecodeError:
               return jsonify({"error": "Не удалось декодировать файл"}), 400
       
       if len(file_content) < 100:
           return jsonify({"error": "Файл слишком мал для пакетного обучения"}), 400
       
       # Получаем параметры
       session_id = request.form.get('session_id', str(uuid.uuid4()))
       chunk_size = int(request.form.get('chunk_size', 500))
       importance = float(request.form.get('importance', 0.7))
       
       # Запускаем обработку файла
       result = assistant.process_bulk_file(
           file_content=file_content,
           filename=file.filename,
           session_id=session_id,
           chunk_size=chunk_size,
           importance=importance
       )
       
       return jsonify(result)
       
   except Exception as e:
       return jsonify({"error": str(e)}), 500

@app.route('/api/bulk/task/<task_id>')
def get_bulk_task_status(task_id):
   """Получение статуса задачи пакетного обучения."""
   try:
       result = assistant.get_bulk_task_status(task_id)
       return jsonify(result)
   except Exception as e:
       return jsonify({"error": str(e)}), 500

@app.route('/api/bulk/progress')
def get_bulk_progress():
   """Получение лога прогресса пакетного обучения."""
   try:
       limit = request.args.get('limit', 50, type=int)
       progress_log = assistant.get_bulk_progress_log(limit)
       return jsonify({
           "success": True,
           "progress": progress_log,
           "total_entries": len(progress_log)
       })
   except Exception as e:
       return jsonify({"error": str(e)}), 500

@app.route('/api/bulk/tasks/<session_id>')
def get_session_bulk_tasks(session_id):
   """Получение всех задач пакетного обучения для сессии."""
   try:
       session_tasks = {}
       for task_id, task_info in assistant.bulk_tasks.items():
           if task_info.get('session_id') == session_id:
               session_tasks[task_id] = {
                   'status': task_info['status'],
                   'start_time': task_info['start_time'].isoformat(),
                   'filename': task_info.get('filename', 'Текст'),
               }
               if 'end_time' in task_info:
                   session_tasks[task_id]['end_time'] = task_info['end_time'].isoformat()
               if 'result' in task_info:
                   session_tasks[task_id]['summary'] = {
                       'total_chunks': task_info['result'].get('total_chunks', 0),
                       'processed_chunks': task_info['result'].get('processed_chunks', 0),
                       'processing_time': task_info['result'].get('processing_time_seconds', 0)
                   }
       
       return jsonify({
           "success": True,
           "tasks": session_tasks
       })
   except Exception as e:
       return jsonify({"error": str(e)}), 500

@app.route('/api/bulk/export/<task_id>')
def export_bulk_task_report(task_id):
   """Экспорт отчета о задаче пакетного обучения."""
   try:
       if task_id not in assistant.bulk_tasks:
           return jsonify({"error": "Задача не найдена"}), 404
       
       task_info = assistant.bulk_tasks[task_id]
       
       # Создаем временный файл с отчетом
       report = {
           "task_id": task_id,
           "export_time": datetime.now().isoformat(),
           "task_info": {
               "status": task_info['status'],
               "start_time": task_info['start_time'].isoformat(),
               "session_id": task_info['session_id']
           }
       }
       
       if 'end_time' in task_info:
           report["task_info"]["end_time"] = task_info['end_time'].isoformat()
       
       if 'result' in task_info:
           report["result"] = task_info['result']
       
       if 'error' in task_info:
           report["error"] = task_info['error']
       
       # Создаем временный файл
       temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
       json.dump(report, temp_file, indent=2, ensure_ascii=False)
       temp_file.close()
       
       filename = f"bulk_learning_report_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
       
       return send_file(
           temp_file.name,
           as_attachment=True,
           download_name=filename,
           mimetype='application/json'
       )
       
   except Exception as e:
       return jsonify({"error": str(e)}), 500

@app.route('/api/bulk/cleanup', methods=['POST'])
def cleanup_bulk_tasks():
   """Очистка старых задач пакетного обучения."""
   try:
       data = request.get_json() or {}
       max_age_hours = data.get('max_age_hours', 24)
       
       removed_count = assistant.cleanup_old_tasks(max_age_hours)
       
       return jsonify({
           "success": True,
           "removed_tasks": removed_count,
           "remaining_tasks": len(assistant.bulk_tasks)
       })
       
   except Exception as e:
       return jsonify({"error": str(e)}), 500

@app.route('/api/knowledge/<session_id>')
def get_knowledge(session_id):
   """Получение знаний для вкладки 'Знания'."""
   try:
       knowledge = assistant.get_session_knowledge(session_id)
       return jsonify(knowledge)
   except Exception as e:
       return jsonify({"error": str(e)}), 500

@app.route('/api/system')
def get_system_status():
   """Получение системного статуса."""
   try:
       status = assistant.get_system_status()
       return jsonify(status)
   except Exception as e:
       return jsonify({"error": str(e)}), 500

@app.route('/api/sessions/<session_id>')
def get_session_info(session_id):
   """Получение информации о сессии."""
   try:
       session = assistant.get_or_create_session(session_id)
       stats = assistant._get_session_stats(session)
       return jsonify({
           "session": stats,
           "messages": session["messages"][-50:]  # Последние 50 сообщений
       })
   except Exception as e:
       return jsonify({"error": str(e)}), 500

@app.route('/api/health')
def health_check():
   """Проверка здоровья системы."""
   return jsonify({
       "status": "healthy" if assistant.global_stats["system_healthy"] else "degraded",
       "neurograph_available": NEUROGRAPH_AVAILABLE,
       "bulk_learning_available": BULK_LEARNING_AVAILABLE,
       "active_sessions": len(assistant.user_sessions),
       "active_bulk_tasks": len(assistant.bulk_tasks) if hasattr(assistant, 'bulk_tasks') else 0,
       "uptime": str(datetime.now() - assistant.global_stats["start_time"]).split('.')[0]
   })

if __name__ == '__main__':
   # Создаем папку templates если не существует
   if not os.path.exists('templates'):
       os.makedirs('templates')
   
   print("🚀 Запуск NeuroGraph Web Assistant")
   print("=" * 50)
   print(f"NeuroGraph доступен: {NEUROGRAPH_AVAILABLE}")
   print(f"Пакетное обучение доступно: {BULK_LEARNING_AVAILABLE}")
   print(f"Активных сессий: {len(assistant.user_sessions)}")
   print("Веб-интерфейс будет доступен по адресу: http://localhost:5000")
   print("=" * 50)
   
   app.run(debug=True, host='0.0.0.0', port=5000)