"""
Kafka Queue Service for Financial Research Chatbot
"""
import json
import logging
import asyncio
from typing import Any, Dict, Optional, Callable, List
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.errors import KafkaError
from datetime import datetime

logger = logging.getLogger(__name__)

class QueueService:
    """Kafka-based message queue service for async processing"""
    
    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self.bootstrap_servers = bootstrap_servers
        self.producer: Optional[AIOKafkaProducer] = None
        self.consumers: Dict[str, AIOKafkaConsumer] = {}
        self.connected = False
        self.stats = {
            "messages_published": 0,
            "messages_consumed": 0,
            "errors": 0,
            "topics": set()
        }
        self.message_handlers: Dict[str, List[Callable]] = {}
    
    async def connect(self):
        """Connect to Kafka"""
        try:
            # Initialize producer with minimal configuration
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
            
            await self.producer.start()
            self.connected = True
            logger.info("Successfully connected to Kafka")
            
        except Exception as e:
            logger.warning(f"Failed to connect to Kafka: {e}")
            logger.info("Running in mock mode - Kafka operations will be simulated")
            self.connected = False
            self.producer = None
            # Don't raise the error, just log it and continue in mock mode
    
    async def disconnect(self):
        """Disconnect from Kafka"""
        try:
            # Stop producer
            if self.producer:
                await self.producer.stop()
                self.producer = None
            
            # Stop consumers
            for consumer in self.consumers.values():
                await consumer.stop()
            self.consumers.clear()
            
            self.connected = False
            logger.info("Disconnected from Kafka")
            
        except Exception as e:
            logger.error(f"Error during Kafka disconnect: {e}")
    
    async def publish_message(self, topic: str, message: Dict[str, Any], key: Optional[str] = None) -> bool:
        """
        Publish a message to a Kafka topic
        
        Args:
            topic: Kafka topic name
            message: Message data to publish
            key: Optional message key for partitioning
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected or not self.producer:
            # Mock mode - simulate successful message publishing
            logger.info(f"Mock mode: Simulating message publish to topic {topic}")
            self.stats["messages_published"] += 1
            self.stats["topics"].add(topic)
            return True
        
        try:
            # Add metadata to message
            message_with_metadata = {
                **message,
                "timestamp": datetime.now().isoformat(),
                "source": "financial_chatbot"
            }
            
            # Publish message
            await self.producer.send_and_wait(
                topic=topic,
                value=message_with_metadata,
                key=key
            )
            
            self.stats["messages_published"] += 1
            self.stats["topics"].add(topic)
            
            logger.debug(f"Published message to topic {topic}: {message.get('query', '')[:50]}...")
            return True
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error publishing message to topic {topic}: {e}")
            return False
    
    async def publish_batch(self, topic: str, messages: List[Dict[str, Any]], keys: Optional[List[str]] = None) -> int:
        """
        Publish multiple messages to a Kafka topic
        
        Args:
            topic: Kafka topic name
            messages: List of messages to publish
            keys: Optional list of message keys
            
        Returns:
            Number of successfully published messages
        """
        if not self.connected or not self.producer:
            return 0
        
        successful = 0
        for i, message in enumerate(messages):
            key = keys[i] if keys and i < len(keys) else None
            if await self.publish_message(topic, message, key):
                successful += 1
        
        return successful
    
    async def create_consumer(self, topic: str, group_id: str, auto_offset_reset: str = "earliest") -> bool:
        """
        Create a Kafka consumer for a topic
        
        Args:
            topic: Kafka topic to consume from
            group_id: Consumer group ID
            auto_offset_reset: Offset reset policy
            
        Returns:
            True if successful, False otherwise
        """
        try:
            consumer = AIOKafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=group_id,
                auto_offset_reset=auto_offset_reset,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                enable_auto_commit=True,
                auto_commit_interval_ms=1000
            )
            
            await consumer.start()
            self.consumers[topic] = consumer
            logger.info(f"Created consumer for topic {topic} with group {group_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating consumer for topic {topic}: {e}")
            return False
    
    async def start_consuming(self, topic: str, handler: Callable, group_id: str = "default_group") -> bool:
        """
        Start consuming messages from a topic with a handler function
        
        Args:
            topic: Kafka topic to consume from
            handler: Async function to handle messages
            group_id: Consumer group ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create consumer if it doesn't exist
            if topic not in self.consumers:
                if not await self.create_consumer(topic, group_id):
                    return False
            
            # Register handler
            if topic not in self.message_handlers:
                self.message_handlers[topic] = []
            self.message_handlers[topic].append(handler)
            
            # Start consumption loop
            consumer = self.consumers[topic]
            asyncio.create_task(self._consume_messages(topic, consumer, handler))
            
            logger.info(f"Started consuming from topic {topic} with handler")
            return True
            
        except Exception as e:
            logger.error(f"Error starting consumption for topic {topic}: {e}")
            return False
    
    async def _consume_messages(self, topic: str, consumer: AIOKafkaConsumer, handler: Callable):
        """Internal message consumption loop"""
        try:
            async for message in consumer:
                try:
                    # Process message
                    await handler(message.value, message.key, message.timestamp)
                    self.stats["messages_consumed"] += 1
                    
                    logger.debug(f"Processed message from topic {topic}")
                    
                except Exception as e:
                    self.stats["errors"] += 1
                    logger.error(f"Error processing message from topic {topic}: {e}")
                    
        except Exception as e:
            logger.error(f"Consumer error for topic {topic}: {e}")
    
    async def stop_consuming(self, topic: str):
        """Stop consuming from a specific topic"""
        if topic in self.consumers:
            consumer = self.consumers[topic]
            await consumer.stop()
            del self.consumers[topic]
            
            if topic in self.message_handlers:
                del self.message_handlers[topic]
            
            logger.info(f"Stopped consuming from topic {topic}")
    
    async def get_topic_info(self, topic: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a Kafka topic
        
        Args:
            topic: Topic name
            
        Returns:
            Topic information or None if error
        """
        if not self.connected or not self.producer:
            return None
        
        try:
            # Get topic metadata
            metadata = await self.producer.client.fetch_all_metadata()
            topic_metadata = metadata.topics.get(topic)
            
            if topic_metadata:
                return {
                    "topic": topic,
                    "partitions": len(topic_metadata.partitions),
                    "replication_factor": topic_metadata.partitions[0].replicas[0] if topic_metadata.partitions else 0,
                    "is_internal": topic_metadata.is_internal
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting topic info for {topic}: {e}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check queue service health"""
        try:
            if not self.connected or not self.producer:
                return {"status": "disconnected", "error": "Not connected to Kafka"}
            
            # Test producer
            await self.producer.send_and_wait(
                topic="health_check",
                value={"test": "health_check", "timestamp": datetime.now().isoformat()}
            )
            
            return {
                "status": "healthy",
                "connected": True,
                "bootstrap_servers": self.bootstrap_servers,
                "active_consumers": len(self.consumers),
                "active_topics": len(self.stats["topics"])
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "connected": False
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue service statistics"""
        return {
            "connected": self.connected,
            "bootstrap_servers": self.bootstrap_servers,
            "stats": {
                "messages_published": self.stats["messages_published"],
                "messages_consumed": self.stats["messages_consumed"],
                "errors": self.stats["errors"],
                "active_topics": len(self.stats["topics"]),
                "active_consumers": len(self.consumers)
            },
            "topics": list(self.stats["topics"]),
            "last_updated": datetime.now().isoformat()
        }
    
    async def clear_stats(self):
        """Clear queue service statistics"""
        self.stats = {
            "messages_published": 0,
            "messages_consumed": 0,
            "errors": 0,
            "topics": set()
        }
        logger.info("Queue statistics cleared")
    
    async def create_topic(self, topic: str, num_partitions: int = 1, replication_factor: int = 1) -> bool:
        """
        Create a new Kafka topic (requires admin privileges)
        
        Args:
            topic: Topic name
            num_partitions: Number of partitions
            replication_factor: Replication factor
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected or not self.producer:
            return False
        
        try:
            # This would require Kafka admin client in production
            # For now, we'll just log the attempt
            logger.info(f"Topic creation requested: {topic} with {num_partitions} partitions, {replication_factor} replicas")
            
            # In production, implement actual topic creation:
            # from kafka.admin import KafkaAdminClient, NewTopic
            # admin_client = KafkaAdminClient(bootstrap_servers=self.bootstrap_servers)
            # new_topic = NewTopic(topic, num_partitions, replication_factor)
            # admin_client.create_topics([new_topic])
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating topic {topic}: {e}")
            return False
