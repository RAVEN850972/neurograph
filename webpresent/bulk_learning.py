import os
import json
import time
import uuid
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import threading
import queue

class TextChunker:
    """Разбиение текста на смысловые части."""
    
    def __init__(self, 
                 chunk_size: int = 500, 
                 overlap: int = 50,
                 respect_sentences: bool = True,
                 respect_paragraphs: bool = True):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.respect_sentences = respect_sentences
        self.respect_paragraphs = respect_paragraphs
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Разбивает текст на смысловые части."""
        if not text.strip():
            return []
        
        chunks = []
        metadata = metadata or {}
        
        if self.respect_paragraphs:
            # Сначала разбиваем по параграфам
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            for para_idx, paragraph in enumerate(paragraphs):
                para_chunks = self._chunk_paragraph(paragraph, para_idx, len(paragraphs))
                
                for chunk_data in para_chunks:
                    chunk_data['metadata'].update(metadata)
                    chunks.append(chunk_data)
        else:
            # Простое разбиение по размеру
            simple_chunks = self._simple_chunk(text)
            for i, chunk_text in enumerate(simple_chunks):
                chunks.append({
                    'content': chunk_text,
                    'chunk_id': str(uuid.uuid4()),
                    'chunk_index': i,
                    'total_chunks': len(simple_chunks),
                    'metadata': {**metadata, 'chunk_type': 'simple'}
                })
        
        return chunks
    
    def _chunk_paragraph(self, paragraph: str, para_idx: int, total_paras: int) -> List[Dict[str, Any]]:
        """Разбивает параграф на части."""
        chunks = []
        
        if len(paragraph) <= self.chunk_size:
            # Параграф помещается целиком
            chunks.append({
                'content': paragraph,
                'chunk_id': str(uuid.uuid4()),
                'chunk_index': 0,
                'paragraph_index': para_idx,
                'total_paragraphs': total_paras,
                'metadata': {'chunk_type': 'paragraph', 'is_complete_paragraph': True}
            })
        else:
            # Разбиваем параграф на части
            if self.respect_sentences:
                sentences = self._split_sentences(paragraph)
                sentence_chunks = self._group_sentences(sentences)
            else:
                sentence_chunks = self._simple_chunk(paragraph)
            
            for i, chunk_text in enumerate(sentence_chunks):
                chunks.append({
                    'content': chunk_text,
                    'chunk_id': str(uuid.uuid4()),
                    'chunk_index': i,
                    'paragraph_index': para_idx,
                    'total_paragraphs': total_paras,
                    'total_chunks_in_paragraph': len(sentence_chunks),
                    'metadata': {'chunk_type': 'paragraph_part', 'is_complete_paragraph': False}
                })
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Разбивает текст на предложения."""
        import re
        
        # Простой алгоритм разбиения на предложения
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _group_sentences(self, sentences: List[str]) -> List[str]:
        """Группирует предложения в части нужного размера."""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Добавляем текущую часть
                chunks.append(' '.join(current_chunk))
                
                # Начинаем новую часть с перекрытием
                if self.overlap > 0 and len(current_chunk) > 1:
                    # Берем последнее предложение для перекрытия
                    current_chunk = [current_chunk[-1], sentence]
                    current_length = len(current_chunk[-2]) + sentence_length
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Добавляем последнюю часть
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _simple_chunk(self, text: str) -> List[str]:
        """Простое разбиение по размеру символов."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Ищем ближайший пробел для корректного разрыва
            if end < len(text):
                while end > start and text[end] != ' ':
                    end -= 1
                
                if end == start:  # Если не нашли пробел
                    end = start + self.chunk_size
            
            chunks.append(text[start:end])
            start = end - self.overlap if self.overlap > 0 else end
        
        return chunks


class BulkLearningSystem:
    """Система пакетного обучения NeuroGraph на больших объемах данных."""
    
    def __init__(self, neurograph_engine, 
                 batch_size: int = 10,
                 delay_between_batches: float = 0.5,
                 progress_callback: Optional[Callable] = None):
        self.engine = neurograph_engine
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
        self.progress_callback = progress_callback
        
        self.chunker = TextChunker()
        self.learning_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'start_time': None,
            'end_time': None,
            'errors': []
        }
        
        # Для асинхронной обработки
        self._processing_queue = queue.Queue()
        self._results_queue = queue.Queue()
        self._is_processing = False
        self._worker_thread = None
    
    def process_text_file(self, file_path: str, 
                         encoding: str = 'utf-8',
                         category: str = 'document',
                         importance: float = 0.7) -> Dict[str, Any]:
        """Обрабатывает текстовый файл."""
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                text = f.read()
            
            metadata = {
                'source_file': os.path.basename(file_path),
                'source_path': file_path,
                'file_size': len(text),
                'category': category,
                'processed_at': datetime.now().isoformat()
            }
            
            return self.process_large_text(text, metadata, importance)
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'file_path': file_path
            }
    
    def process_multiple_files(self, file_paths: List[str], 
                              encoding: str = 'utf-8',
                              category: str = 'documents') -> Dict[str, Any]:
        """Обрабатывает несколько файлов."""
        
        results = {
            'total_files': len(file_paths),
            'processed_files': 0,
            'failed_files': 0,
            'results': [],
            'errors': []
        }
        
        for file_path in file_paths:
            try:
                self._report_progress(f"Обработка файла: {os.path.basename(file_path)}")
                
                result = self.process_text_file(file_path, encoding, category)
                results['results'].append(result)
                
                if result.get('success', False):
                    results['processed_files'] += 1
                else:
                    results['failed_files'] += 1
                    results['errors'].append({
                        'file': file_path,
                        'error': result.get('error', 'Unknown error')
                    })
                
                # Небольшая пауза между файлами
                time.sleep(self.delay_between_batches)
                
            except Exception as e:
                results['failed_files'] += 1
                results['errors'].append({
                    'file': file_path,
                    'error': str(e)
                })
        
        return results
    
    def process_large_text(self, text: str, 
                          metadata: Dict[str, Any] = None,
                          importance: float = 0.7) -> Dict[str, Any]:
        """Обрабатывает большой текст, разбивая его на части."""
        
        self.learning_stats['start_time'] = datetime.now()
        metadata = metadata or {}
        
        try:
            # Разбиваем текст на части
            chunks = self.chunker.chunk_text(text, metadata)
            
            self._report_progress(f"Текст разбит на {len(chunks)} частей")
            
            # Обрабатываем части пакетами
            results = self._process_chunks_in_batches(chunks, importance)
            
            self.learning_stats['end_time'] = datetime.now()
            processing_time = (self.learning_stats['end_time'] - self.learning_stats['start_time']).total_seconds()
            
            return {
                'success': True,
                'total_chunks': len(chunks),
                'processed_chunks': results['successful'],
                'failed_chunks': results['failed'],
                'processing_time_seconds': processing_time,
                'chunks_per_second': len(chunks) / processing_time if processing_time > 0 else 0,
                'stats': self.learning_stats,
                'sample_chunks': chunks[:3] if chunks else []  # Первые 3 части для примера
            }
            
        except Exception as e:
            self.learning_stats['end_time'] = datetime.now()
            return {
                'success': False,
                'error': str(e),
                'stats': self.learning_stats
            }
    
    def _process_chunks_in_batches(self, chunks: List[Dict], importance: float) -> Dict[str, int]:
        """Обрабатывает части текста пакетами."""
        
        results = {'successful': 0, 'failed': 0}
        
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
            
            self._report_progress(f"Обработка пакета {batch_num}/{total_batches} ({len(batch)} частей)")
            
            # Обрабатываем пакет
            batch_results = self._process_batch(batch, importance)
            results['successful'] += batch_results['successful']
            results['failed'] += batch_results['failed']
            
            # Пауза между пакетами
            if i + self.batch_size < len(chunks):
                time.sleep(self.delay_between_batches)
        
        return results
    
    def _process_batch(self, batch: List[Dict], importance: float) -> Dict[str, int]:
        """Обрабатывает один пакет частей."""
        
        results = {'successful': 0, 'failed': 0}
        
        for chunk_data in batch:
            try:
                # Формируем контент для обучения
                content = chunk_data['content']
                
                # Добавляем контекстную информацию
                context_info = []
                if 'paragraph_index' in chunk_data:
                    context_info.append(f"Параграф {chunk_data['paragraph_index'] + 1}")
                if 'chunk_index' in chunk_data:
                    context_info.append(f"Часть {chunk_data['chunk_index'] + 1}")
                
                if context_info:
                    content = f"[{', '.join(context_info)}] {content}"
                
                # Определяем источник
                source = chunk_data['metadata'].get('source_file', 'bulk_text')
                
                # Определяем теги
                tags = ['bulk_learning']
                if 'category' in chunk_data['metadata']:
                    tags.append(chunk_data['metadata']['category'])
                if 'chunk_type' in chunk_data['metadata']:
                    tags.append(chunk_data['metadata']['chunk_type'])
                
                # Обучаем NeuroGraph
                response = self.engine.learn(content)
                
                if hasattr(response, 'success') and response.success:
                    results['successful'] += 1
                    self.learning_stats['successful'] += 1
                else:
                    results['failed'] += 1
                    self.learning_stats['failed'] += 1
                    error_msg = getattr(response, 'error_message', 'Unknown error')
                    self.learning_stats['errors'].append({
                        'chunk_id': chunk_data['chunk_id'],
                        'error': error_msg
                    })
                
                self.learning_stats['total_processed'] += 1
                
            except Exception as e:
                results['failed'] += 1
                self.learning_stats['failed'] += 1
                self.learning_stats['total_processed'] += 1
                self.learning_stats['errors'].append({
                    'chunk_id': chunk_data.get('chunk_id', 'unknown'),
                    'error': str(e)
                })
        
        return results
    
    def _report_progress(self, message: str):
        """Сообщает о прогрессе."""
        if self.progress_callback:
            self.progress_callback(message)
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    def process_directory(self, directory_path: str, 
                         file_extensions: List[str] = ['.txt', '.md'],
                         recursive: bool = True,
                         encoding: str = 'utf-8') -> Dict[str, Any]:
        """Обрабатывает все файлы в директории."""
        
        directory = Path(directory_path)
        if not directory.exists():
            return {
                'success': False,
                'error': f"Директория не найдена: {directory_path}"
            }
        
        # Находим все файлы
        files = []
        if recursive:
            for ext in file_extensions:
                files.extend(directory.rglob(f"*{ext}"))
        else:
            for ext in file_extensions:
                files.extend(directory.glob(f"*{ext}"))
        
        file_paths = [str(f) for f in files]
        
        self._report_progress(f"Найдено {len(file_paths)} файлов для обработки")
        
        if not file_paths:
            return {
                'success': False,
                'error': 'Файлы для обработки не найдены'
            }
        
        return self.process_multiple_files(file_paths, encoding, 'directory_files')
    
    def export_learning_report(self, output_file: str) -> bool:
        """Экспортирует отчет об обучении."""
        
        try:
            report = {
                'learning_session': {
                    'start_time': self.learning_stats['start_time'].isoformat() if self.learning_stats['start_time'] else None,
                    'end_time': self.learning_stats['end_time'].isoformat() if self.learning_stats['end_time'] else None,
                    'duration_seconds': (
                        self.learning_stats['end_time'] - self.learning_stats['start_time']
                    ).total_seconds() if self.learning_stats['start_time'] and self.learning_stats['end_time'] else None
                },
                'statistics': self.learning_stats,
                'configuration': {
                    'batch_size': self.batch_size,
                    'delay_between_batches': self.delay_between_batches,
                    'chunk_size': self.chunker.chunk_size,
                    'overlap': self.chunker.overlap,
                    'respect_sentences': self.chunker.respect_sentences,
                    'respect_paragraphs': self.chunker.respect_paragraphs
                }
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            self._report_progress(f"Ошибка при экспорте отчета: {e}")
            return False


class AsyncBulkLearning:
    """Асинхронная версия пакетного обучения."""
    
    def __init__(self, neurograph_engine, max_concurrent: int = 3):
        self.engine = neurograph_engine
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    async def process_large_text_async(self, text: str, 
                                     metadata: Dict[str, Any] = None,
                                     chunk_size: int = 500,
                                     importance: float = 0.7) -> Dict[str, Any]:
        """Асинхронная обработка большого текста."""
        
        chunker = TextChunker(chunk_size=chunk_size)
        chunks = chunker.chunk_text(text, metadata or {})
        
        # Создаем задачи для обработки частей
        tasks = []
        for chunk_data in chunks:
            task = self._process_chunk_async(chunk_data, importance)
            tasks.append(task)
        
        # Выполняем задачи
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Подсчитываем результаты
        successful = sum(1 for r in results if r is True)
        failed = len(results) - successful
        
        return {
            'success': True,
            'total_chunks': len(chunks),
            'successful': successful,
            'failed': failed,
            'results': results
        }
    
    async def _process_chunk_async(self, chunk_data: Dict, importance: float) -> bool:
        """Асинхронная обработка одной части."""
        
        async with self.semaphore:
            try:
                # В реальности здесь был бы await для асинхронного NeuroGraph API
                # Пока делаем синхронный вызов в executor
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, 
                    self.engine.learn, 
                    chunk_data['content']
                )
                response = self.engine.learn(content)
                print(f"[DEBUG] Обучение части {chunk_data['chunk_id']}: success={hasattr(response, 'success') and response.success}, response={response}")
                return hasattr(response, 'success') and response.success
                
            except Exception as e:
                print(f"Ошибка обработки части {chunk_data.get('chunk_id', 'unknown')}: {e}")
                return False


# Пример использования
def example_usage():
    """Пример использования системы пакетного обучения."""
    
    try:
        from neurograph.integration import create_default_engine
        
        # Создаем движок
        engine = create_default_engine()
        
        # Создаем систему пакетного обучения
        bulk_system = BulkLearningSystem(
            neurograph_engine=engine,
            batch_size=5,  # Обрабатываем по 5 частей за раз
            delay_between_batches=0.2  # Пауза 200мс между пакетами
        )
        
        # Пример 1: Обработка большого текста
        large_text = """
        Искусственный интеллект (ИИ) — это область компьютерных наук, которая занимается созданием машин, способных выполнять задачи, обычно требующие человеческого интеллекта.
        
        История ИИ началась в 1950-х годах, когда ученые впервые начали серьезно изучать возможность создания думающих машин. Алан Тьюринг предложил знаменитый тест Тьюринга для определения способности машины демонстрировать интеллектуальное поведение.
        
        Современный ИИ включает в себя множество подходов, включая машинное обучение, глубокое обучение, нейронные сети и символическое рассуждение. Машинное обучение позволяет компьютерам учиться на данных без явного программирования.
        
        Применения ИИ включают распознавание речи, компьютерное зрение, обработку естественного языка, робототехнику и многое другое. ИИ уже используется в поисковых системах, рекомендательных системах, автономных транспортных средствах и медицинской диагностике.
        
        Будущее ИИ обещает еще более революционные изменения в том, как мы работаем, общаемся и живем. Однако это также поднимает важные вопросы этики, безопасности и влияния на общество.
        """ * 3  # Увеличиваем текст в 3 раза
        
        print("🚀 Обработка большого текста...")
        result = bulk_system.process_large_text(
            text=large_text,
            metadata={'topic': 'AI', 'source': 'example'},
            importance=0.8
        )
        
        print(f"✅ Результат обработки:")
        print(f"  Всего частей: {result['total_chunks']}")
        print(f"  Успешно: {result['processed_chunks']}")
        print(f"  Ошибок: {result['failed_chunks']}")
        print(f"  Время: {result['processing_time_seconds']:.2f} сек")
        print(f"  Скорость: {result['chunks_per_second']:.2f} частей/сек")
        
        # Пример 2: Создание тестового файла и его обработка
        test_file = "test_document.txt"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(large_text)
        
        print(f"\n📄 Обработка файла {test_file}...")
        file_result = bulk_system.process_text_file(test_file, category='test_document')
        
        if file_result['success']:
            print(f"✅ Файл успешно обработан!")
        else:
            print(f"❌ Ошибка обработки файла: {file_result['error']}")
        
        # Экспорт отчета
        bulk_system.export_learning_report("learning_report.json")
        print("📊 Отчет об обучении сохранен в learning_report.json")
        
        # Очистка
        if os.path.exists(test_file):
            os.remove(test_file)
        
        # Завершение
        engine.shutdown()
        print("✅ Пример завершен!")
        
    except ImportError:
        print("❌ NeuroGraph не установлен. Запустите с установленным NeuroGraph.")
    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    example_usage()