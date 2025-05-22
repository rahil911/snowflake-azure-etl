#!/usr/bin/env python3
"""
ChannelLoader implementation for the Dim_Channel dimension
"""
from sqlalchemy import Table
from ..schemas.dimension.channel import DimChannel
from ..schemas.dimension_loader import DimensionLoader

class ChannelLoader(DimensionLoader):
    """Loader for the Channel dimension"""
    
    def load(self):
        """Load data from staging tables to the Channel dimension"""
        print("\nLoading Dim_Channel table...")
        
        # Ensure Unknown channel exists
        unknown_channel = DimChannel(
            DimChannelID=1,
            ChannelID=-1,
            ChannelCategoryID=-1,
            ChannelName='Unknown Channel',
            ChannelCategory='Unknown'
        )
        print("DEBUG: Ensuring Unknown channel record exists...")
        self.ensure_unknown_record(DimChannel, unknown_channel)
        
        # Reflect staging tables to get actual column names (just for logging)
        print("DEBUG: Reflecting staging tables...")
        staging_channel = Table('STAGING_CHANNEL', self.staging_metadata, autoload_with=self.staging_engine)
        staging_channel_category = Table('STAGING_CHANNELCATEGORY', self.staging_metadata, autoload_with=self.staging_engine)
        
        print(f"Channel table columns: {[c.name for c in staging_channel.columns]}")
        print(f"Channel category table columns: {[c.name for c in staging_channel_category.columns]}")
        
        # Process channels using text-based query
        print("DEBUG: Attempting to load channels...")
        channels_added = 0
        
        # Using text query to avoid case sensitivity issues
        channels = self.execute_text_query("""
            SELECT 
                c.channelid AS ChannelID,
                c.channelcategoryid AS ChannelCategoryID,
                c.channel AS ChannelName,
                cc.channelcategory AS ChannelCategory
            FROM STAGING_CHANNEL c
            JOIN STAGING_CHANNELCATEGORY cc
            ON c.channelcategoryid = cc.channelcategoryid
        """)
        
        print(f"DEBUG: Found {len(channels)} channels")
        
        # Process each channel
        for channel in channels:
            try:
                # Access fields using dictionary-like access to be safe
                channel_id = channel.ChannelID if hasattr(channel, 'ChannelID') else channel[0]
                channel_category_id = channel.ChannelCategoryID if hasattr(channel, 'ChannelCategoryID') else channel[1]
                channel_name = channel.ChannelName if hasattr(channel, 'ChannelName') else channel[2]
                channel_category = channel.ChannelCategory if hasattr(channel, 'ChannelCategory') else channel[3]
                
                # Check if channel exists
                existing = self.session.query(DimChannel).filter(
                    DimChannel.ChannelID == channel_id
                ).first()
                
                if not existing:
                    self.session.add(DimChannel(
                        ChannelID=channel_id,
                        ChannelCategoryID=channel_category_id,
                        ChannelName=channel_name,
                        ChannelCategory=channel_category
                    ))
                    channels_added += 1
            except Exception as e:
                print(f"DEBUG ERROR: Error adding channel: {e}")
        
        print(f"DEBUG: Committing {channels_added} channels")
        self.commit_records()
        
        channel_count = self.get_row_count('DIM_CHANNEL')
        print(f"Loaded {channel_count} channels into Dim_Channel")
        return channel_count 