import { Module } from '@nestjs/common';
import { QueryController } from './query.controller';
import { QueryService } from './query.service';
import { QueryController } from './query.controller';

@Module({
  controllers: [QueryController],
  providers: [QueryService]
})
export class QueryModule {}
